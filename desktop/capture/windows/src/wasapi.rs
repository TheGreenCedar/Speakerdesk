use crate::protocol::{ProducerClock, Source, TICKS_PER_SECOND, render_refill_frames};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::json;
use std::{io::Write, mem::size_of, ptr};
use windows::Win32::{
    Media::Audio::{
        AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
        AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_LOOPBACK,
        AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, DEVICE_STATE_ACTIVE, IAudioCaptureClient,
        IAudioClient, IAudioClock, IAudioRenderClient, IMMDevice, IMMDeviceEnumerator,
        MMDeviceEnumerator, WAVEFORMATEX, eCapture, eConsole, eRender,
    },
    System::{
        Com::{
            CLSCTX_ALL, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
            CoUninitialize,
        },
        Performance::{QueryPerformanceCounter, QueryPerformanceFrequency},
    },
};
use windows::core::Interface;

fn failure(action: &str, error: windows::core::Error) -> String {
    format!("{action}: HRESULT 0x{:08x}: {error}", error.code().0 as u32)
}

pub struct Apartment;
impl Apartment {
    pub fn initialize() -> Result<Self, String> {
        // All endpoint/client/service calls stay on this one STA owner.
        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED)
                .ok()
                .map_err(|error| failure("Initialize Windows capture COM", error))?;
        }
        Ok(Self)
    }
}
impl Drop for Apartment {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}

pub fn now() -> Result<u64, String> {
    let mut counter = 0;
    let mut frequency = 0;
    unsafe {
        QueryPerformanceCounter(&mut counter)
            .map_err(|error| failure("Read capture performance counter", error))?;
        QueryPerformanceFrequency(&mut frequency)
            .map_err(|error| failure("Read capture performance frequency", error))?;
    }
    if counter < 0 || frequency <= 0 {
        return Err("Windows returned an invalid capture performance clock.".into());
    }
    u64::try_from(counter as u128 * TICKS_PER_SECOND as u128 / frequency as u128)
        .map_err(|_| "Capture performance clock overflow.".into())
}

#[derive(Clone, PartialEq, Serialize)]
pub struct MixFormat {
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    block_align: u16,
    format_bytes_base64: String,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Route {
    endpoint_id: String,
    mix_format: MixFormat,
}

fn endpoint_id(device: &IMMDevice) -> Result<String, String> {
    unsafe {
        let id = device
            .GetId()
            .map_err(|error| failure("Read capture endpoint identity", error))?;
        let text = id
            .to_string()
            .map_err(|error| format!("Decode capture endpoint identity: {error}"));
        CoTaskMemFree(Some(id.0.cast()));
        text
    }
}
fn mix_format(client: &IAudioClient) -> Result<MixFormat, String> {
    unsafe {
        let pointer = client
            .GetMixFormat()
            .map_err(|error| failure("Read endpoint mix format", error))?;
        if pointer.is_null() {
            return Err("WASAPI returned no endpoint mix format.".into());
        }
        let format = pointer.read_unaligned();
        let extra = format.cbSize as usize;
        let value = if extra > 4096 || format.nSamplesPerSec == 0 || format.nChannels == 0 {
            Err("WASAPI returned an invalid endpoint mix format.".into())
        } else {
            let bytes =
                std::slice::from_raw_parts(pointer.cast::<u8>(), size_of::<WAVEFORMATEX>() + extra);
            Ok(MixFormat {
                tag: format.wFormatTag,
                channels: format.nChannels,
                sample_rate: format.nSamplesPerSec,
                bits_per_sample: format.wBitsPerSample,
                block_align: format.nBlockAlign,
                format_bytes_base64: STANDARD.encode(bytes),
            })
        };
        CoTaskMemFree(Some(pointer.cast()));
        value
    }
}

pub struct Packet {
    pub samples: Vec<f32>,
    pub ticks: u64,
    pub device_position: u64,
    pub error_after: Option<String>,
}

// Keep an idle render endpoint producing real loopback packets and timestamps.
struct SilentRender {
    client: IAudioClient,
    render: IAudioRenderClient,
    capacity: u32,
    running: bool,
}
impl SilentRender {
    fn open(device: &IMMDevice, format: &WAVEFORMATEX) -> Result<Self, String> {
        unsafe {
            let client: IAudioClient = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|error| failure("Activate silent loopback render client", error))?;
            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                    1_000_000,
                    0,
                    format,
                    None,
                )
                .map_err(|error| failure("Initialize silent loopback render stream", error))?;
            let capacity = client
                .GetBufferSize()
                .map_err(|error| failure("Read silent render capacity", error))?;
            render_refill_frames(capacity, 0, false)?;
            let render: IAudioRenderClient = client
                .GetService::<IAudioRenderClient>()
                .map_err(|error| failure("Get silent render service", error))?;
            Ok(Self {
                client,
                render,
                capacity,
                running: false,
            })
        }
    }
    fn refill(&mut self) -> Result<(), String> {
        unsafe {
            let padding = self
                .client
                .GetCurrentPadding()
                .map_err(|error| failure("Read silent render padding", error))?;
            let frames = render_refill_frames(self.capacity, padding, self.running)?;
            if frames != 0 {
                self.render
                    .GetBuffer(frames)
                    .map_err(|error| failure("Acquire silent render buffer", error))?;
                // SILENT makes the engine ignore the returned buffer contents.
                self.render
                    .ReleaseBuffer(frames, AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)
                    .map_err(|error| failure("Release silent render buffer", error))?;
            }
            Ok(())
        }
    }
    fn start(&mut self, resume: bool) -> Result<(), String> {
        unsafe {
            if resume {
                self.client
                    .Reset()
                    .map_err(|error| failure("Reset paused silent render stream", error))?;
            }
            self.refill()?;
            self.client
                .Start()
                .map_err(|error| failure("Start silent loopback render stream", error))?;
        }
        self.running = true;
        Ok(())
    }
    fn stop(&mut self) -> Result<(), String> {
        if self.running {
            unsafe { self.client.Stop() }
                .map_err(|error| failure("Stop silent loopback render stream", error))?;
            self.running = false;
        }
        Ok(())
    }
}
impl Drop for SilentRender {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

pub struct Endpoint {
    pub source: Source,
    pub route: Route,
    device: IMMDevice,
    client: IAudioClient,
    capture: IAudioCaptureClient,
    clock: IAudioClock,
    silent_render: Option<SilentRender>,
    capacity: u32,
    running: bool,
    seen_packet: bool,
    diagnostics: bool,
    clock_diagnostics: u8,
    packet_diagnosed: bool,
}
impl Endpoint {
    pub fn open(source: Source) -> Result<Self, String> {
        unsafe {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .map_err(|error| failure("Create endpoint enumerator", error))?;
            let device=enumerator.GetDefaultAudioEndpoint(if source==Source::Microphone {eCapture} else {eRender},eConsole)
                .map_err(|error|failure(&format!("Open default {} endpoint; check Windows sound devices and microphone privacy",source.name()),error))?;
            if device
                .GetState()
                .map_err(|error| failure("Read capture device state", error))?
                != DEVICE_STATE_ACTIVE
            {
                return Err(format!(
                    "The default {} endpoint is inactive.",
                    source.name()
                ));
            }
            let client: IAudioClient = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|error| failure("Activate WASAPI capture client", error))?;
            let route = Route {
                endpoint_id: endpoint_id(&device)?,
                mix_format: mix_format(&client)?,
            };
            let channels = source.channels() as u16;
            let format = WAVEFORMATEX {
                wFormatTag: 3,
                nChannels: channels,
                nSamplesPerSec: 16000,
                nAvgBytesPerSec: 16000 * u32::from(channels) * 4,
                nBlockAlign: channels * 4,
                wBitsPerSample: 32,
                cbSize: 0,
            };
            let flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
                | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
                | if source == Source::System {
                    AUDCLNT_STREAMFLAGS_LOOPBACK
                } else {
                    0
                };
            client.Initialize(AUDCLNT_SHAREMODE_SHARED,flags,1_000_000,0,&format,None)
                .map_err(|error|failure(&format!("Initialize {} shared Float32 16000 Hz/{channels} channels with automatic channel/sample-rate conversion (mix {} Hz/{} channels); select a compatible Windows audio endpoint",source.name(),route.mix_format.sample_rate,route.mix_format.channels),error))?;
            let capacity = client
                .GetBufferSize()
                .map_err(|error| failure("Read WASAPI capture capacity", error))?;
            if capacity == 0 || capacity > 16000 {
                return Err(format!(
                    "WASAPI capture buffer exceeds its bounded one-second contract: {capacity} frames."
                ));
            }
            let capture: IAudioCaptureClient = client
                .GetService::<IAudioCaptureClient>()
                .map_err(|error| failure("Get WASAPI capture service", error))?;
            let clock: IAudioClock = client
                .GetService::<IAudioClock>()
                .map_err(|error| failure("Get WASAPI producer clock", error))?;
            let silent_render = if source == Source::System {
                // Reuse this exact IMMDevice, rather than resolving a second default route.
                Some(SilentRender::open(&device, &format)?)
            } else {
                None
            };
            Ok(Self {
                source,
                route,
                device,
                client,
                capture,
                clock,
                silent_render,
                capacity,
                running: false,
                seen_packet: false,
                diagnostics: std::env::var("SPEAKERDESK_WASAPI_DIAGNOSTICS")
                    .is_ok_and(|value| value == "1"),
                clock_diagnostics: 0,
                packet_diagnosed: false,
            })
        }
    }
    pub fn verify_route(&self) -> Result<(), String> {
        unsafe {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .map_err(|error| failure("Recheck endpoint enumerator", error))?;
            let device = enumerator
                .GetDefaultAudioEndpoint(
                    if self.source == Source::Microphone {
                        eCapture
                    } else {
                        eRender
                    },
                    eConsole,
                )
                .map_err(|error| failure("Recheck default capture endpoint", error))?;
            let client: IAudioClient = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|error| failure("Recheck endpoint format", error))?;
            let observed = Route {
                endpoint_id: endpoint_id(&device)?,
                mix_format: mix_format(&client)?,
            };
            if observed != self.route
                || device
                    .GetState()
                    .map_err(|error| failure("Recheck capture device state", error))?
                    != DEVICE_STATE_ACTIVE
            {
                return Err(format!(
                    "The {} endpoint or mix format changed; restart recording. Original audio is retained.",
                    self.source.name()
                ));
            }
            Ok(())
        }
    }
    pub fn start(&mut self, resume: bool) -> Result<(), String> {
        self.verify_route()?;
        unsafe {
            if resume {
                self.client
                    .Reset()
                    .map_err(|error| failure("Reset paused WASAPI capture stream", error))?;
            }
            self.client.Start().map_err(|error| {
                failure(
                    "Start WASAPI capture; check endpoint availability and microphone privacy",
                    error,
                )
            })?;
        }
        self.running = true;
        self.seen_packet = false;
        if let Some(render) = &mut self.silent_render {
            if let Err(error) = render.start(resume) {
                return Err(match self.stop() {
                    Ok(()) => error,
                    Err(cleanup) => format!("{error}; {cleanup}"),
                });
            }
        }
        Ok(())
    }
    pub fn stop(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();
        if self.running {
            match unsafe { self.client.Stop() } {
                Ok(()) => self.running = false,
                Err(error) => errors.push(failure("Stop WASAPI capture stream", error)),
            }
        }
        if let Some(render) = &mut self.silent_render {
            if let Err(error) = render.stop() {
                errors.push(error);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
    pub fn observe(&mut self) -> Result<ProducerClock, String> {
        if let Some(render) = &mut self.silent_render {
            render.refill()?;
        }
        unsafe {
            if self
                .device
                .GetState()
                .map_err(|error| failure("Observe capture device state", error))?
                != DEVICE_STATE_ACTIVE
            {
                return Err(format!(
                    "The {} endpoint became inactive.",
                    self.source.name()
                ));
            }
            let mut position = 0;
            let mut ticks = 0;
            // Retain S_OK/S_FALSE and the raw outputs for readiness diagnostics;
            // the generated Result<()> binding otherwise erases successful HRESULTs.
            let status = (Interface::vtable(&self.clock).GetPosition)(
                Interface::as_raw(&self.clock),
                &mut position,
                &mut ticks,
            );
            if self.diagnostics && self.clock_diagnostics == 0 {
                self.clock_diagnostics += 1;
                eprintln!(
                    "{}",
                    json!({"type":"wasapi_clock_diagnostic","source":self.source.name(),
                    "route":self.route,"seen_packet":self.seen_packet,
                    "get_position":{"hresult":format!("0x{:08x}",status.0 as u32),"device_position":position,"qpc_100ns":ticks},
                    "silent_render_capacity_frames":self.silent_render.as_ref().map(|render|render.capacity)})
                );
                let _ = std::io::stderr().flush();
            }
            status.ok().map_err(|error| failure(
                &format!("Observe {} WASAPI producer clock (device_position={position}, qpc_100ns={ticks})", self.source.name()), error
            ))?;
            Ok(ProducerClock {
                hresult: status.0,
                position,
                ticks,
            })
        }
    }
    pub fn next(&mut self) -> Result<Option<Packet>, String> {
        unsafe {
            if self
                .capture
                .GetNextPacketSize()
                .map_err(|error| failure("Poll WASAPI capture producer", error))?
                == 0
            {
                return Ok(None);
            }
            let mut data = ptr::null_mut();
            let mut frames = 0;
            let mut flags = 0;
            let mut position = 0;
            let mut ticks = 0;
            let status = (Interface::vtable(&self.capture).GetBuffer)(
                Interface::as_raw(&self.capture),
                &mut data,
                &mut frames,
                &mut flags,
                &mut position,
                &mut ticks,
            );
            if self.diagnostics && !self.packet_diagnosed && (frames != 0 || status.is_err()) {
                self.packet_diagnosed = true;
                eprintln!(
                    "{}",
                    json!({"type":"wasapi_packet_diagnostic","source":self.source.name(),
                    "get_buffer_hresult":format!("0x{:08x}",status.0 as u32),
                    "frames":frames,"flags":flags,"device_position_frames":position,"qpc_100ns":ticks})
                );
            }
            status
                .ok()
                .map_err(|error| failure("Read WASAPI capture packet", error))?;
            if frames == 0 {
                return Ok(None);
            }
            let copied = (|| {
                if frames > self.capacity
                    || frames > 16000
                    || ticks == 0
                    || flags & !7 != 0
                    || flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32 != 0
                {
                    return Err(format!(
                        "{} capture returned an untrusted timestamp/buffer: frames={frames}, flags={flags}, qpc={ticks}; this packet was not admitted.",
                        self.source.name()
                    ));
                }
                let count = frames as usize * self.source.channels();
                let samples = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                    vec![0.0; count]
                } else {
                    if data.is_null() {
                        return Err("WASAPI returned a null nonsilent capture buffer.".into());
                    }
                    std::slice::from_raw_parts(data, count * 4)
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|bytes| f32::from_le_bytes(*bytes))
                        .collect::<Vec<_>>()
                };
                if samples.iter().any(|value| !value.is_finite()) {
                    return Err("WASAPI returned nonfinite Float32 capture PCM; this packet was not admitted.".into());
                }
                let discontinuity = flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0
                    && self.seen_packet;
                Ok(Packet {samples,ticks,device_position:position,error_after:discontinuity.then(||format!("{} capture reported a device discontinuity; recording stopped and accepted audio is retained.",self.source.name()))})
            })();
            let release = self
                .capture
                .ReleaseBuffer(frames)
                .map_err(|error| failure("Release WASAPI capture packet", error));
            match copied {
                Ok(mut packet) => {
                    if let Err(error) = release {
                        packet.error_after = Some(error);
                    }
                    self.seen_packet = true;
                    Ok(Some(packet))
                }
                Err(error) => Err(error),
            }
        }
    }
}
impl Drop for Endpoint {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

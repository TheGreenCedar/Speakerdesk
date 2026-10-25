use crate::{
    output::{Failure, Output},
    protocol::{self, AudioTimeline, Command, ProducerClock, Progress, RecordingClock, Source},
    wasapi::{self, Endpoint},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, RecvTimeoutError},
    time::{Duration, Instant},
};

pub enum Input {
    Command(Command),
    Error(String),
    Eof,
}
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Starting,
    Recording,
    Paused,
    Resuming,
}
struct Producer {
    endpoint: Endpoint,
    timeline: AudioTimeline,
    packet_seen: bool,
    observation: ProducerClock,
    device_position: u64,
}
struct Session {
    producers: Vec<Producer>,
    clock: RecordingClock,
    floor: u64,
    phase: Phase,
    pending: VecDeque<Value>,
    progress: Progress,
    ready_deadline: Instant,
    next_fence: Instant,
}
impl Session {
    fn new(microphone: bool, system: bool) -> Result<Self, String> {
        let mut producers = Vec::new();
        for (source, enabled) in [(Source::Microphone, microphone), (Source::System, system)] {
            if enabled {
                producers.push(Producer {
                    endpoint: Endpoint::open(source)?,
                    timeline: AudioTimeline::default(),
                    packet_seen: false,
                    observation: ProducerClock::default(),
                    device_position: 0,
                });
            }
        }
        let floor = wasapi::now()?;
        for producer in &mut producers {
            producer.endpoint.start(false)?;
        }
        Ok(Self {
            producers,
            clock: RecordingClock::new(floor),
            floor,
            phase: Phase::Starting,
            pending: VecDeque::new(),
            progress: Progress::default(),
            ready_deadline: Instant::now() + Duration::from_secs(5),
            next_fence: Instant::now(),
        })
    }
    fn stop_producers(&mut self) -> Result<(), String> {
        let errors: Vec<_> = self
            .producers
            .iter_mut()
            .filter_map(|producer| producer.endpoint.stop().err())
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
    fn route_event(&self, phase: &str) -> Value {
        let sources:Vec<_>=self.producers.iter().map(|producer|json!({"source":producer.endpoint.source.name(),
            "route":producer.endpoint.route,"processing_observation":producer.endpoint.processing,
            "capture_format":{"rate":16000,"channels":producer.endpoint.source.channels(),"format":"float32_le"},
            "observation_kind":if producer.packet_seen {"packet_and_clock"} else {"clock_and_empty_buffer"},
            "producer_qpc_100ns":producer.observation.ticks,"device_position":producer.device_position,
            "producer_clock_hresult":producer.observation.hresult,"producer_clock_device_position":producer.observation.position})).collect();
        json!({"type":"capture_runtime","state_phase":phase,"capability_version":protocol::CAPABILITY,
            "export_contract":protocol::EXPORT,"progress_contract":protocol::PROGRESS,
            "processing":"original_sources_no_apple_processing","sources":sources})
    }
    fn packets(&mut self, output: &Output) -> Result<(), Failure> {
        for producer in &mut self.producers {
            for index in 0..128 {
                let Some(mut packet) = producer.endpoint.next()? else {
                    break;
                };
                let channels = producer.endpoint.source.channels();
                let frames = packet.samples.len() / channels;
                let trim = if packet.ticks < self.floor {
                    (self.floor - packet.ticks).div_ceil(625) as usize
                } else {
                    0
                };
                if trim >= frames {
                    continue;
                }
                if trim > 0 {
                    packet.samples.drain(..trim * channels);
                    packet.ticks += trim as u64 * 625;
                }
                let time = producer.timeline.accept(
                    packet.ticks,
                    packet.samples.len() / channels,
                    &self.clock,
                )?;
                producer.packet_seen = true;
                producer.device_position = packet.device_position;
                let events =
                    protocol::audio_events(producer.endpoint.source, time, &packet.samples)?;
                if matches!(self.phase, Phase::Starting | Phase::Resuming) {
                    self.pending.extend(events);
                    if self.pending.len() > 32 {
                        return Err(Failure {message:"Capture reached its bounded startup queue before every producer was ready.".into(),retained:std::mem::take(&mut self.pending)});
                    }
                } else {
                    output.batch(events)?;
                }
                if let Some(error) = packet.error_after {
                    return Err(error.into());
                }
                if index == 127 {
                    return Err("WASAPI did not drain within its bounded producer pass."
                        .to_string()
                        .into());
                }
            }
        }
        Ok(())
    }
    fn poll(&mut self, output: &Output) -> Result<(), Failure> {
        let ingress = wasapi::now()?;
        for producer in &mut self.producers {
            producer.observation = producer.endpoint.observe()?;
            producer.device_position = producer.observation.position;
        }
        self.packets(output)?;
        // Preserve observe-then-drain fence ordering. A missing startup clock
        // remains untrusted while the bounded readiness wait gathers packet evidence.
        for producer in &self.producers {
            if matches!(self.phase, Phase::Starting | Phase::Resuming)
                && producer.observation.pending()
            {
                continue;
            }
            producer
                .observation
                .validate()
                .map_err(|error| format!("{} capture: {error}", producer.endpoint.source.name()))?;
        }
        let ready = self.producers.iter().all(|producer| {
            producer
                .observation
                .ready(producer.endpoint.source, self.floor, producer.packet_seen)
        });
        if matches!(self.phase, Phase::Starting | Phase::Resuming) {
            if !ready {
                if Instant::now() >= self.ready_deadline {
                    let observations = self.producers.iter().map(|producer| format!(
                        "{}: HRESULT 0x{:08x}, device_position={}, qpc_100ns={}, packet_seen={}",
                        producer.endpoint.source.name(), producer.observation.hresult as u32,
                        producer.observation.position, producer.observation.ticks, producer.packet_seen,
                    )).collect::<Vec<_>>().join("; ");
                    return Err(format!("Capture did not observe fresh microphone/loopback producer state within five seconds; check Windows audio devices and microphone privacy. Last producer observations: {observations}").into());
                }
                return Ok(());
            }
            for producer in &self.producers {
                producer.endpoint.verify_route()?;
            }
            let phase = if self.phase == Phase::Starting {
                "after_start"
            } else {
                "after_resume"
            };
            let mut events = VecDeque::from([
                self.route_event(phase),
                json!({"type":"recording","microphone":self.producers.iter().any(|producer|producer.endpoint.source==Source::Microphone),
                "system":self.producers.iter().any(|producer|producer.endpoint.source==Source::System),"time":self.clock.at(ingress)}),
            ]);
            events.append(&mut self.pending);
            output.batch(events)?;
            self.phase = Phase::Recording;
        }
        if Instant::now() >= self.next_fence {
            for producer in &self.producers {
                producer.endpoint.verify_route()?;
            }
            let observed = self
                .producers
                .iter()
                .map(|producer| producer.observation.ticks)
                .min()
                .unwrap_or(ingress)
                .min(ingress);
            output.event(self.progress.fence(self.clock.at(observed), ready)?)?;
            self.next_fence = Instant::now() + Duration::from_millis(250);
        }
        Ok(())
    }
    fn pause(&mut self, output: &Output) -> Result<(), Failure> {
        if self.phase == Phase::Paused {
            return Ok(());
        }
        self.stop_producers()?;
        self.packets(output)?;
        let now = wasapi::now()?;
        self.clock.pause(now);
        let mut events = std::mem::take(&mut self.pending);
        events.push_back(json!({"type":"paused","time":self.clock.at(now)}));
        output.batch(events)?;
        self.phase = Phase::Paused;
        Ok(())
    }
    fn resume(&mut self) -> Result<(), String> {
        if self.phase != Phase::Paused {
            return Ok(());
        }
        for producer in &self.producers {
            producer.endpoint.verify_route()?;
        }
        let floor = wasapi::now()?;
        for producer in &mut self.producers {
            producer.endpoint.start(true)?;
            producer.timeline = AudioTimeline::default();
            producer.packet_seen = false;
            producer.observation = ProducerClock::default();
        }
        self.clock.resume(floor)?;
        self.floor = floor;
        self.phase = Phase::Resuming;
        self.ready_deadline = Instant::now() + Duration::from_secs(5);
        self.next_fence = Instant::now();
        Ok(())
    }
    fn stop(&mut self, output: &Output) -> Result<(), Failure> {
        self.stop_producers()?;
        if self.phase != Phase::Paused {
            self.packets(output)?;
        }
        let mut events = std::mem::take(&mut self.pending);
        events.push_back(json!({"type":"stopped","time":self.clock.at(wasapi::now()?)}));
        output.batch(events)
    }
}

pub fn run(commands: Receiver<Input>) -> Result<(), String> {
    let _apartment = wasapi::Apartment::initialize()?;
    let output = Output::new();
    let mut session: Option<Session> = None;
    let result = (|| -> Result<(), Failure> {
        loop {
            let wait = if session
                .as_ref()
                .is_some_and(|session| session.phase != Phase::Paused)
            {
                Duration::from_millis(5)
            } else {
                Duration::from_secs(1)
            };
            match commands.recv_timeout(wait) {
                Ok(Input::Command(Command::Start {
                    microphone,
                    system,
                    job_id,
                    capture_export_contract,
                })) => {
                    if session.is_some() {
                        return Err("Capture is already started.".to_string().into());
                    }
                    protocol::validate_start(
                        microphone,
                        system,
                        job_id.as_deref(),
                        capture_export_contract.as_deref(),
                    )?;
                    session = Some(Session::new(microphone, system)?);
                }
                Ok(Input::Command(Command::Pause)) => {
                    if let Some(session) = &mut session {
                        session.pause(&output)?;
                    }
                }
                Ok(Input::Command(Command::Resume)) => {
                    if let Some(session) = &mut session {
                        session.resume()?;
                    }
                }
                Ok(Input::Command(Command::Stop))
                | Ok(Input::Eof)
                | Err(RecvTimeoutError::Disconnected) => {
                    if let Some(session) = &mut session {
                        session.stop(&output)?;
                    } else {
                        output.event(json!({"type":"stopped","time":0.0}))?;
                    }
                    return Ok(());
                }
                Ok(Input::Error(error)) => return Err(error.into()),
                Err(RecvTimeoutError::Timeout) => (),
            }
            if let Some(session) = &mut session {
                if session.phase != Phase::Paused {
                    session.poll(&output)?;
                }
            }
        }
    })();
    if let Err(mut failure) = result {
        if let Some(session) = &mut session {
            if let Err(error) = session.stop_producers() {
                failure.message.push_str(&format!("; {error}"));
            }
            let mut pending = std::mem::take(&mut session.pending);
            pending.append(&mut failure.retained);
            failure.retained = pending;
        }
        drop(session);
        failure.retained.push_back(json!({"type":"error","error":failure.message,"startup_phase":"windows_wasapi_capture"}));
        if let Err(error) = output.drain(failure.retained) {
            eprintln!("{error}");
        }
        if let Err(error) = output.finish() {
            eprintln!("{error}");
        }
        return Err(failure.message);
    }
    drop(session);
    output.finish()
}

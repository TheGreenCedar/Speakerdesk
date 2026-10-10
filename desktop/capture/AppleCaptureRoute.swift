import Foundation
import AVFoundation
import AudioToolbox
import CryptoKit

let appleCaptureCapability = "apple_input_node_output_route_v1"
let appleCaptureExport = "apple_processed_mono16k_v1"

func captureFormat(_ format: AVAudioFormat) -> [String:Any] {
    let layout: String
    if format.channelCount == 1 { layout = "mono" }
    else if format.channelCount == 9 && format.commonFormat == .pcmFormatFloat32 && !format.isInterleaved &&
        format.channelLayout?.layoutTag == (kAudioChannelLayoutTag_DiscreteInOrder|9) {
        layout = "discrete_float32_noninterleaved"
    } else { layout = "unsupported" }
    return ["sample_rate":Int(format.sampleRate),"channels":Int(format.channelCount),"layout":layout]
}

func captureFormatMatches(_ a: AVAudioFormat, _ b: AVAudioFormat) -> Bool {
    a.sampleRate == b.sampleRate && a.channelCount == b.channelCount &&
        a.commonFormat == b.commonFormat && a.isInterleaved == b.isInterleaved &&
        a.channelLayout?.layoutTag == b.channelLayout?.layoutTag
}

// The microphone tap and render bus have different roles. VoiceProcessingIO
// may expose replicated microphone channels alongside a stereo render bus.
// Keep exact microphone/callback matching; admit only the explicit stereo
// Float32 render layout in addition to the previously identical I/O format.
func appleStereoRenderFormat(_ format: AVAudioFormat) -> Bool {
    let tag = format.channelLayout?.layoutTag
    return format.channelCount == 2 && format.commonFormat == .pcmFormatFloat32 &&
        !format.isInterleaved && (tag == nil || tag == kAudioChannelLayoutTag_Stereo)
}

func renderFormat(_ format: AVAudioFormat) -> [String:Any] {
    if appleStereoRenderFormat(format) {
        return ["sample_rate":Int(format.sampleRate),"channels":2,"layout":"stereo_float32_noninterleaved"]
    }
    return captureFormat(format)
}

func captureFormatDetails(_ format: AVAudioFormat) -> [String:Any] {
    ["sample_rate":format.sampleRate,"channels":Int(format.channelCount),
     "common_format":Int(format.commonFormat.rawValue),"interleaved":format.isInterleaved,
     "channel_layout_tag":Int(format.channelLayout?.layoutTag ?? 0)]
}

func appleRenderMatches(_ microphone: AVAudioFormat, _ render: AVAudioFormat) -> Bool {
    captureFormatMatches(microphone,render) ||
        (microphone.sampleRate == render.sampleRate && appleStereoRenderFormat(render))
}

func ownedCaptureHash() throws -> String {
    let url=URL(fileURLWithPath:CommandLine.arguments[0]).standardizedFileURL
    return SHA256.hash(data:try Data(contentsOf:url)).map{String(format:"%02x",$0)}.joined()
}

// No semantic channel selection or amplitude gate. Silence is valid.
func appleMicrophoneCopy(_ input:AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
    if input.format.channelCount == 9 { return try collapseExactDiscreteReplicas(input) }
    guard input.format.channelCount == 1, input.format.commonFormat == .pcmFormatFloat32,
          !input.format.isInterleaved, let pointer=input.floatChannelData?[0], input.frameLength > 0,
          (0..<Int(input.frameLength)).allSatisfy({pointer[$0].isFinite}) else {
        throw NSError(domain:"AppleCaptureRoute",code:1,userInfo:[NSLocalizedDescriptionKey:"Unsupported microphone output channels; no channel selected."])
    }
    let copy=AVAudioPCMBuffer(pcmFormat:input.format,frameCapacity:input.frameLength)!
    copy.frameLength=input.frameLength
    memcpy(copy.floatChannelData![0],pointer,Int(input.frameLength)*4)
    return copy
}

func appleRoutePacket(jobID:String, helperHash:String, input:AVAudioFormat, output:AVAudioFormat,
                      callback:AVAudioFormat, readbacks:[String:Bool]) throws -> [String:Any] {
    let checks:[(String,Bool)] = [
        ("render_format",appleRenderMatches(input,output)),
        ("microphone_callback_format",captureFormatMatches(input,callback)),
        ("input_enabled",readbacks["input_enabled"] == true),
        ("output_enabled",readbacks["output_enabled"] == true),
        ("engine_running",readbacks["engine_running"] == true),
        ("bypass_off",readbacks["bypassed"] == false),
        ("manual_rendering_off",readbacks["manual_rendering"] == false),
        ("input_unmuted",readbacks["input_muted"] == false)]
    let failures=checks.filter{!$0.1}.map{$0.0}
    guard failures.isEmpty else {
        throw NSError(domain:"AppleCaptureRoute",code:2,userInfo:[
            NSLocalizedDescriptionKey:"Apple microphone runtime state or negotiated I/O format is unsupported.",
            "failed_checks":failures,
            "capture_observation":["input_output":captureFormat(input),
                                   "output_input":renderFormat(output),
                                   "callback":captureFormat(callback),"readbacks":readbacks,
                                   "input_output_details":captureFormatDetails(input),
                                   "output_input_details":captureFormatDetails(output),
                                   "callback_details":captureFormatDetails(callback)]])
    }
    let format=captureFormat(callback)
    guard format["layout"] as? String != "unsupported" else {throw NSError(domain:"AppleCaptureRoute",code:3)}
    return ["type":"capture_processing","schema_version":1,"job_id":jobID,"helper_sha256":helperHash,
            "capability_version":appleCaptureCapability,"processor":"apple_voice_processing_io",
            "export_contract":appleCaptureExport,"output":"processed_microphone","stage":"retained_source_pcm",
            "postprocessing":"identity","stream_epoch":0,
            "format":["sample_rate":16000,"channels":1,"layout":"mono"],
            "route":["node":"inputNode","bus":0,"tap_scope":"output"],"state_phase":"after_start",
            "channel_mapping":callback.channelCount == 1 ? "mono_direct" : "nine_discrete_exact_replicas",
            "callback_format":format,"io_formats":["input_output":captureFormat(input),"output_input":renderFormat(output)],
            "readbacks":readbacks]
}

// Immutable observation belongs to a copied tap callback, not its later drain.
struct AppleCaptureObservation {
    let input: AVAudioFormat
    let output: AVAudioFormat
    let readbacks: [String:Bool]
    func packet(jobID:String,helperHash:String,callback:AVAudioFormat) throws -> [String:Any] {
        try appleRoutePacket(jobID:jobID,helperHash:helperHash,input:input,output:output,
                             callback:callback,readbacks:readbacks)
    }
}
func observeAppleCapture(_ engine:AVAudioEngine) -> AppleCaptureObservation {
    AppleCaptureObservation(input:engine.inputNode.outputFormat(forBus:0),
        output:engine.outputNode.inputFormat(forBus:0),readbacks:[
            "input_enabled":engine.inputNode.isVoiceProcessingEnabled,
            "output_enabled":engine.outputNode.isVoiceProcessingEnabled,
            "engine_running":engine.isRunning,"bypassed":engine.inputNode.isVoiceProcessingBypassed,
            "manual_rendering":engine.isInManualRenderingMode,"input_muted":engine.inputNode.isVoiceProcessingInputMuted,
            "agc_enabled":engine.inputNode.isVoiceProcessingAGCEnabled])
}

// System audio must not outrun the first validated processed-microphone tap.
// AVAudioEngine.start returning is not a microphone callback acknowledgement.
enum AppleMicrophoneStartupState { case waiting, ready, stopped }

func startSystemAfterAppleMicrophone(timeout: Double = 10,
    readiness: () -> AppleMicrophoneStartupState,
    startSystem: () async throws -> Void) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + timeout
    while true {
        try Task.checkCancellation()
        switch readiness() {
        case .ready:
            try await startSystem()
            try Task.checkCancellation()
            guard case .ready = readiness() else {
                throw NSError(domain:"AppleCaptureRoute",code:11,userInfo:[NSLocalizedDescriptionKey:
                    "Microphone startup was interrupted while Mac audio capture was starting."])
            }
            return
        case .stopped:
            throw NSError(domain:"AppleCaptureRoute",code:11,userInfo:[NSLocalizedDescriptionKey:
                "Microphone startup was interrupted before Mac audio capture began."])
        case .waiting:
            guard ProcessInfo.processInfo.systemUptime < deadline else {
                throw NSError(domain:"AppleCaptureRoute",code:10,userInfo:[NSLocalizedDescriptionKey:
                    "The processed microphone did not deliver a supported callback within 10 seconds. Mac audio capture did not start."])
            }
            try await Task.sleep(nanoseconds:20_000_000)
        }
    }
}

// Failure and command paths join one cleanup operation. Keeping the engine
// active until that owner drains avoids dropping accepted copied callbacks.
actor AppleCaptureStopOwner {
    private var operation: Task<Void, Never>?

    func run(_ cleanup: @escaping () async -> Void) async {
        if let operation {await operation.value;return}
        let owned=Task {await cleanup()}
        operation=owned
        await owned.value
    }

    func reset() async {
        if let operation {await operation.value}
        operation=nil
    }
}

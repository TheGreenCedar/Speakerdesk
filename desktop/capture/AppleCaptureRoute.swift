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
    guard captureFormatMatches(input,output),captureFormatMatches(input,callback),
          readbacks["input_enabled"] == true, readbacks["output_enabled"] == true,
          readbacks["engine_running"] == true,readbacks["bypassed"] == false,
          readbacks["manual_rendering"] == false,readbacks["input_muted"] == false else {
        throw NSError(domain:"AppleCaptureRoute",code:2,userInfo:[NSLocalizedDescriptionKey:"Apple microphone runtime state or negotiated I/O format is unsupported."])
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
            "callback_format":format,"io_formats":["input_output":captureFormat(input),"output_input":captureFormat(output)],
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

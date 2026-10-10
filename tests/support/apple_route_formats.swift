import Foundation
import AVFoundation
import AudioToolbox

@main struct AppleRouteFormatChecks {
    static func main() throws {
        let mono=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,channels:1,interleaved:false)!
        let stereo=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,channels:2,interleaved:false)!
        let layout=AVAudioChannelLayout(layoutTag:kAudioChannelLayoutTag_DiscreteInOrder|9)!
        let nine=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,interleaved:false,channelLayout:layout)
        let state=["input_enabled":true,"output_enabled":true,"engine_running":true,"bypassed":false,
                   "manual_rendering":false,"input_muted":false,"agc_enabled":false]
        var checks:[String:Bool]=[:]
        let job=String(repeating:"a",count:32),helper=String(repeating:"b",count:64)
        func packet(_ mic:AVAudioFormat,_ render:AVAudioFormat,_ callback:AVAudioFormat,
                    _ readbacks:[String:Bool]) throws -> [String:Any] {
            try appleRoutePacket(jobID:job,helperHash:helper,input:mic,output:render,callback:callback,readbacks:readbacks)
        }
        precondition(!captureFormatMatches(nine,stereo))
        checks["old_equality_rejects_reported_nine_vs_stereo"]=true
        for (label,mic,render) in [("mono",mono,mono),("nine",nine,nine),
                                   ("mono_stereo",mono,stereo),("nine_stereo",nine,stereo)] {
            let receipt=try packet(mic,render,mic,state)
            precondition(receipt["job_id"] as? String==job)
            let formats=receipt["io_formats"] as! [String:[String:Any]]
            precondition(formats["input_output"]!["channels"] as? Int==Int(mic.channelCount))
            if render===stereo {precondition(formats["output_input"]!["layout"] as? String=="stereo_float32_noninterleaved")}
            checks[label+"_receipt"]=true
        }
        func refuse(_ label:String,_ mic:AVAudioFormat,_ render:AVAudioFormat,_ callback:AVAudioFormat,
                    _ readbacks:[String:Bool],_ predicate:String) {
            do {_=try packet(mic,render,callback,readbacks);preconditionFailure(label)}
            catch {
                let value=error as NSError
                precondition(value.domain=="AppleCaptureRoute" && value.code==2)
                precondition((value.userInfo["failed_checks"] as? [String])?.contains(predicate)==true)
                precondition(value.userInfo["capture_observation"] != nil)
                checks[label]=true
            }
        }
        let rate=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:44100,channels:2,interleaved:false)!
        let interleaved=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,channels:2,interleaved:true)!
        let integer=AVAudioFormat(commonFormat:.pcmFormatInt16,sampleRate:48000,channels:2,interleaved:false)!
        let three=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,interleaved:false,
                               channelLayout:AVAudioChannelLayout(layoutTag:kAudioChannelLayoutTag_DiscreteInOrder|3)!)
        let discrete=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,interleaved:false,
                                  channelLayout:AVAudioChannelLayout(layoutTag:kAudioChannelLayoutTag_DiscreteInOrder|2)!)
        for (label,render) in [("rate_mismatch",rate),("interleaved_render",interleaved),
                               ("integer_render",integer),("unknown_channels",three),("unknown_stereo_layout",discrete)] {
            refuse(label,nine,render,nine,state,"render_format")
        }
        refuse("callback_mismatch",nine,stereo,mono,state,"microphone_callback_format")
        for (key,required,predicate) in [("input_enabled",true,"input_enabled"),("output_enabled",true,"output_enabled"),
              ("engine_running",true,"engine_running"),("bypassed",false,"bypass_off"),
              ("manual_rendering",false,"manual_rendering_off"),("input_muted",false,"input_unmuted")] {
            var bad=state;bad[key] = !required
            refuse("state_"+key,nine,stereo,nine,bad,predicate)
        }
        let pcm=AVAudioPCMBuffer(pcmFormat:nine,frameCapacity:137)!;pcm.frameLength=137
        for channel in 0..<9 {for frame in 0..<137 {pcm.floatChannelData![channel][frame]=Float(frame)/1000}}
        let collapsed=try appleMicrophoneCopy(pcm)
        precondition(memcmp(collapsed.floatChannelData![0],pcm.floatChannelData![0],137*4)==0)
        checks["nine_replicas_lossless_after_stereo_receipt"]=true
        pcm.floatChannelData![8][55] += 0.001
        do {_=try appleMicrophoneCopy(pcm);preconditionFailure("distinct channels")}
        catch {precondition((error as NSError).domain=="LosslessRedundantChannels");checks["distinct_mic_channels_still_refuse"]=true}
        let result:[String:Any]=["checks":checks,"count":checks.count,"device_opened":false,
            "capture_started":false,"neural_calls":0,"nine_stereo_packet":try packet(nine,stereo,nine,state)]
        let data=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys])
        FileHandle.standardOutput.write(data+Data([10]))
    }
}

import Foundation
import AVFoundation
import AudioToolbox

// Lossless channel equivalence only. Route identity comes from the owned
// input-node output and runtime state; distinct channels always refuse.
func collapseExactDiscreteReplicas(_ input:AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
    let channels=Int(input.format.channelCount),frames=Int(input.frameLength)
    guard input.format.commonFormat == .pcmFormatFloat32,
          !input.format.isInterleaved,frames>0,channels==9,
          input.format.channelLayout?.layoutTag == (kAudioChannelLayoutTag_DiscreteInOrder|9) else {
        throw NSError(domain:"LosslessRedundantChannels",code:1,userInfo:[NSLocalizedDescriptionKey:"Unsupported replica layout; no channel selected."])
    }
    let buffers=UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating:input.audioBufferList))
    let bytes=frames*MemoryLayout<Float>.size
    guard buffers.count==channels,let common=buffers[0].mData else {throw NSError(domain:"LosslessRedundantChannels",code:2)}
    for buffer in buffers {
        guard buffer.mNumberChannels==1,Int(buffer.mDataByteSize)>=bytes,
              let data=buffer.mData,memcmp(common,data,bytes)==0 else {
            throw NSError(domain:"LosslessRedundantChannels",code:3,userInfo:[NSLocalizedDescriptionKey:"Input channels differ; lossless replica collapse refused."])
        }
    }
    let samples=common.bindMemory(to:Float.self,capacity:frames)
    guard (0..<frames).allSatisfy({samples[$0].isFinite}) else {throw NSError(domain:"LosslessRedundantChannels",code:5)}
    let mono=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:input.format.sampleRate,channels:1,interleaved:false)!
    let output=AVAudioPCMBuffer(pcmFormat:mono,frameCapacity:input.frameLength)!;output.frameLength=input.frameLength
    memcpy(output.floatChannelData![0],common,bytes)
    guard memcmp(output.floatChannelData![0],common,bytes)==0 else {throw NSError(domain:"LosslessRedundantChannels",code:4)}
    return output
}

import Foundation
import AVFoundation

@main
struct ConversionCheck {
    static func main() throws {
        for (rate, channels) in [(44100.0, 1), (48000.0, 1), (16000.0, 2), (48000.0, 2)] {
            let format = AVAudioFormat(commonFormat:.pcmFormatFloat32, sampleRate:rate, channels:AVAudioChannelCount(channels), interleaved:channels == 2)!
            let target = AVAudioFormat(commonFormat:.pcmFormatFloat32, sampleRate:16000, channels:AVAudioChannelCount(channels), interleaved:channels == 2)!
            let count = Int(rate * 2)
            let samples = (0..<count*channels).map { index in
                let t = Double(index/channels)/rate
                return Float(0.04*sin(2*Double.pi*Double(997+index%channels*312)*t) + 0.03*sin(2*Double.pi*1733*t))
            }
            func buffer(_ start: Int, _ end: Int) -> AVAudioPCMBuffer {
                let pcm = AVAudioPCMBuffer(pcmFormat:format, frameCapacity:AVAudioFrameCount(end-start))!
                pcm.frameLength = pcm.frameCapacity
                samples.withUnsafeBufferPointer { src in
                    pcm.floatChannelData![0].update(from:src.baseAddress!+start*channels,count:(end-start)*channels)
                }
                return pcm
            }
            let converter = try CaptureAudioConverter(input:format, output:target)
            var chunks:[ConvertedAudioChunk] = []
            var pos = 0
            let sizes = [Int(rate/10), 2048, 4096, Int(rate/10), 512]
            var packet = 0
            while pos < count {
                let end = min(count,pos+sizes[packet%sizes.count])
                chunks += try converter.convert(buffer(pos,end), at:0.26375+Double(pos)/rate)
                pos=end;packet+=1
            }
            chunks += try converter.finish()
            let repeatedFinish = try converter.finish()
            assert(repeatedFinish.isEmpty)
            var actual:[Float] = [];var expectedTime=0.26375
            for chunk in chunks {
                assert(abs(chunk.time-expectedTime)<1e-10,"Converted samples overlap or leave a gap")
                let n=Int(chunk.pcm.frameLength)
                actual += Array(UnsafeBufferPointer(start:chunk.pcm.floatChannelData![0],count:n*channels))
                expectedTime += Double(n)/16000
            }
            assert(abs(actual.count/channels-32000)<=1,"SRC tail lost or duplicated")
            let whole = try CaptureAudioConverter(input:format,output:target)
            let reference = try whole.convert(buffer(0,count),at:0.26375)+whole.finish()
            var expected:[Float]=[]
            for chunk in reference { expected += Array(UnsafeBufferPointer(start:chunk.pcm.floatChannelData![0],count:Int(chunk.pcm.frameLength)*channels)) }
            assert(expected.count==actual.count)
            let error=zip(expected,actual).map { abs($0-$1) }.max() ?? 0
            assert(error<1e-6,"Buffered callbacks changed the converted speech waveform")
            // Reproduce the former one-call converter protocol independently.
            let legacy = AVAudioConverter(from:format,to:target)!
            var legacySamples:[Float]=[];var legacyCounts:[Int]=[];pos=0
            while pos<count {
                let end=min(count,pos+Int(rate/10))
                let out:[Float]=autoreleasepool {
                    let input=buffer(pos,end)
                    let capacity=AVAudioFrameCount(ceil(Double(input.frameLength)*16000/rate)+64)
                    let output=AVAudioPCMBuffer(pcmFormat:target,frameCapacity:capacity)!
                    var supplied=false;var error:NSError?
                    legacy.convert(to:output,error:&error) { _, status in
                        if supplied { status.pointee = .noDataNow;return nil }
                        supplied=true;status.pointee = .haveData;return input
                    }
                    assert(error==nil)
                    return Array(UnsafeBufferPointer(start:output.floatChannelData![0],count:Int(output.frameLength)*channels))
                }
                legacySamples += out;legacyCounts.append(out.count);pos=end
            }
            let common=min(expected.count,legacySamples.count)
            let legacyError=zip(expected.prefix(common),legacySamples.prefix(common)).map { abs($0-$1) }.max() ?? 0
            print("LEGACY rate=\(rate) frames=\(legacySamples.count) counts=\(legacyCounts) max_waveform_error=\(legacyError)")
            if rate != 16000 {
                assert(legacySamples.count < expected.count,"Legacy fixture did not reproduce the missing EOF tail")
                assert(legacyCounts.contains { $0 != 1600*channels },"Legacy fixture did not reproduce buffered callback lengths")
            }
            assert(!converter.discontinuous(at:2.26475),"Ordinary callback jitter must not reset the SRC")
            assert(converter.discontinuous(at:3.5))
            print("PASS rate=\(rate) channels=\(channels) frames=\(actual.count/channels) max_waveform_error=\(error) contiguous=true EOF_idempotent=true")
        }
    }
}

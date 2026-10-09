// AVAudioConverter may retain input between callbacks. Its output has one
// continuous sample clock; callback timestamps cannot timestamp buffered output.
import Foundation
import AVFoundation

struct ConvertedAudioTimeline {
    private var origin: Double?
    private var outputFrames: Int64 = 0
    private(set) var expectedInputTime: Double?

    mutating func acceptInput(time: Double, frames: Int, rate: Double) -> Bool {
        // Small callback jitter and clock drift do not tear the output stream.
        // A real discontinuity must finish/reset the converter first.
        let discontinuity = expectedInputTime.map { abs(time - $0) > 0.02 } ?? false
        expectedInputTime = time + Double(frames) / rate
        if origin == nil { origin = time }
        return discontinuity
    }

    mutating func emit(frames: Int) -> Double {
        precondition(frames >= 0 && origin != nil)
        let position = origin! + Double(outputFrames) / 16000
        outputFrames += Int64(frames)
        return position
    }
}


struct ConvertedAudioChunk {
    let time: Double
    let pcm: AVAudioPCMBuffer
}

final class CaptureAudioConverter {
    private let converter: AVAudioConverter
    private var timeline = ConvertedAudioTimeline()
    private var finished = false
    var inputFormat: AVAudioFormat { converter.inputFormat }

    init(input: AVAudioFormat, output: AVAudioFormat) throws {
        guard let value = AVAudioConverter(from: input, to: output) else {
            throw NSError(domain: "CaptureAudioConverter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Audio format conversion failed."])
        }
        converter = value
    }

    func discontinuous(at time: Double) -> Bool {
        timeline.expectedInputTime.map { abs(time - $0) > 0.02 } ?? false
    }

    func convert(_ input: AVAudioPCMBuffer, at time: Double) throws -> [ConvertedAudioChunk] {
        precondition(!finished)
        _ = timeline.acceptInput(time: time, frames: Int(input.frameLength), rate: input.format.sampleRate)
        return try pump(input)
    }

    func finish() throws -> [ConvertedAudioChunk] {
        guard !finished, timeline.expectedInputTime != nil else { return [] }
        finished = true
        return try pump(nil)
    }

    private func pump(_ input: AVAudioPCMBuffer?) throws -> [ConvertedAudioChunk] {
        var supplied = false
        var chunks: [ConvertedAudioChunk] = []
        // Bound every drain, including a malfunctioning platform converter.
        for _ in 0..<32 {
            guard let pcm = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: 4000) else {
                throw NSError(domain: "CaptureAudioConverter", code: 2)
            }
            var error: NSError?
            let status = converter.convert(to: pcm, error: &error) { _, state in
                if let input, !supplied { supplied = true; state.pointee = .haveData; return input }
                state.pointee = input == nil ? .endOfStream : .noDataNow
                return nil
            }
            if let error { throw error }
            if status == .error { throw NSError(domain: "CaptureAudioConverter", code: 3) }
            if pcm.frameLength > 0 {
                chunks.append(ConvertedAudioChunk(time: timeline.emit(frames: Int(pcm.frameLength)), pcm: pcm))
            }
            if status != .haveData { return chunks }
        }
        throw NSError(domain: "CaptureAudioConverter", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "Audio converter did not finish its buffered output."])
    }
}

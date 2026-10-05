// macOS14+ local audio helper. Idle until an explicit JSON start command.
// Audio-only SCStream output; no screen images are read, emitted, or saved.
import Foundation
import AVFoundation
import ScreenCaptureKit
import CoreMedia
import CoreGraphics

final class MeetingCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let output = DispatchQueue(label: "speakerdesk.capture.output")
    private let audioQueue = DispatchQueue(label: "speakerdesk.capture.audio")
    private let stateLock = NSLock()
    private var engine = AVAudioEngine()
    private var stream: SCStream?
    private var converters: [String: AVAudioConverter] = [:]
    private var startHost = 0.0
    private var pausedAt = 0.0
    private var pausedSeconds = 0.0
    private var active = false
    private var paused = false
    private var micEnabled = false
    private var timer: DispatchSourceTimer?
    private let pcmFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
        sampleRate: 16000, channels: 1, interleaved: false)!

    func emit(_ value: [String: Any]) {
        output.async {
            guard let data = try? JSONSerialization.data(withJSONObject: value) else { return }
            FileHandle.standardOutput.write(data + Data([10]))
        }
    }
    func drainOutput() { output.sync {} }
    private func hostSeconds() -> Double { AVAudioTime.seconds(forHostTime: mach_absolute_time()) }
    private func elapsed() -> Double {
        stateLock.withLock { max(0, (paused ? pausedAt : hostSeconds()) - startHost - pausedSeconds) }
    }
    private func packet(_ input: AVAudioPCMBuffer, source: String, timestamp: Double) {
        // The serial audio queue owns both converters; state snapshots use a lock.
        let timing = stateLock.withLock { (active, paused, startHost, pausedSeconds) }
        guard timing.0 && !timing.1 else { return }
        var converter = converters[source]
        if converter == nil || converter!.inputFormat != input.format {
            converter = AVAudioConverter(from: input.format, to: pcmFormat)
            converters[source] = converter
        }
        guard let converter else { emit(["type":"error","error":"Audio format conversion failed."]); return }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength)*16000/input.format.sampleRate)+64)
        guard let pcm = AVAudioPCMBuffer(pcmFormat: pcmFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        converter.convert(to: pcm, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true; status.pointee = .haveData; return input
        }
        if let error { emit(["type":"error","error":error.localizedDescription]); return }
        guard pcm.frameLength > 0, let pointer = pcm.floatChannelData?[0] else { return }
        let audio = Data(bytes: pointer, count: Int(pcm.frameLength)*4)
        let position = max(0, timestamp-timing.2-timing.3)
        emit(["type":"audio","source":source,"time":position,"rate":16000,
              "pcm":audio.base64EncodedString()])
    }
    func start(microphone: Bool, system: Bool) async throws {
        guard !stateLock.withLock({ active }) else { throw NSError(domain:"MeetingCapture",code:1,
            userInfo:[NSLocalizedDescriptionKey:"A meeting is already recording."]) }
        guard microphone || system else { throw NSError(domain:"MeetingCapture",code:2,
            userInfo:[NSLocalizedDescriptionKey:"Select a microphone or Mac audio."]) }
        if microphone {
            let allowed = await AVCaptureDevice.requestAccess(for: .audio)
            if !allowed { throw NSError(domain:"MeetingCapture",code:3,
                userInfo:[NSLocalizedDescriptionKey:"Microphone access was denied. Enable Speakerdesk in System Settings → Privacy & Security → Microphone."]) }
        }
        if system {
            // This call may prompt. It runs only after the user's Start meeting action.
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let display = content.displays.first else {
                throw NSError(domain:"MeetingCapture",code:4,userInfo:[NSLocalizedDescriptionKey:"Mac audio capture needs an available display."])
            }
            let ownApps = content.applications.filter { $0.bundleIdentifier == "com.speakerdesk.desktop" }
            let filter = SCContentFilter(display: display, excludingApplications: ownApps, exceptingWindows: [])
            let config = SCStreamConfiguration()
            config.width = 2; config.height = 2
            config.minimumFrameInterval = CMTime(value:1,timescale:1)
            config.capturesAudio = true; config.excludesCurrentProcessAudio = true
            config.sampleRate = 16000; config.channelCount = 1
            let capture = SCStream(filter: filter, configuration: config, delegate: self)
            try capture.addStreamOutput(self, type:.audio, sampleHandlerQueue: audioQueue)
            stream = capture
        }
        micEnabled = microphone
        stateLock.withLock { startHost = hostSeconds(); pausedSeconds = 0; paused = false; active = true }
        do {
            if let stream { try await stream.startCapture() }
            if microphone {
                let node = engine.inputNode
                let format = node.outputFormat(forBus:0)
                guard format.sampleRate > 0 else { throw NSError(domain:"MeetingCapture",code:5,
                    userInfo:[NSLocalizedDescriptionKey:"No microphone is available."]) }
                node.installTap(onBus:0, bufferSize:2048, format:format) { [weak self] buffer, time in
                    guard let self else { return }
                    // The tap owns the buffer only for this callback; copy before asynchronous conversion.
                    guard let copy = AVAudioPCMBuffer(pcmFormat:buffer.format, frameCapacity:buffer.frameLength) else { return }
                    copy.frameLength = buffer.frameLength
                    let src = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
                    let dst = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
                    for i in 0..<src.count { memcpy(dst[i].mData!, src[i].mData!, Int(src[i].mDataByteSize)) }
                    let timestamp = time.isHostTimeValid ? AVAudioTime.seconds(forHostTime:time.hostTime) : self.hostSeconds()
                    self.audioQueue.async { self.packet(copy,source:"microphone",timestamp:timestamp) }
                }
                engine.prepare(); try engine.start()
            }
            let ticker = DispatchSource.makeTimerSource(queue:output)
            ticker.schedule(deadline:.now(), repeating:.milliseconds(250))
            ticker.setEventHandler { [weak self] in
                guard let self, self.stateLock.withLock({ self.active }) else { return }
                self.emit(["type":"clock","time":self.elapsed(),"paused":self.stateLock.withLock({ self.paused })])
            }
            timer = ticker; ticker.resume()
            emit(["type":"recording","microphone":microphone,"system":system])
        } catch {
            await stop(); throw error
        }
    }
    func pause() async throws {
        let timing = stateLock.withLock { (active, paused, startHost, pausedSeconds) }
        guard timing.0 && !timing.1 else { return }
        if let stream { try await stream.stopCapture() }
        if micEnabled { engine.pause() }
        // Drain copied audio before changing timeline state.
        audioQueue.sync {}
        stateLock.withLock { pausedAt = hostSeconds(); paused = true }
        emit(["type":"paused","time":elapsed()])
    }
    func resume() async throws {
        guard stateLock.withLock({ active && paused }) else { return }
        stateLock.withLock { pausedSeconds += hostSeconds()-pausedAt; paused = false }
        if let stream { try await stream.startCapture() }
        if micEnabled { try engine.start() }
        emit(["type":"recording","time":elapsed()])
    }
    func stop() async {
        guard stateLock.withLock({ active }) else { return }
        timer?.cancel(); timer = nil
        if let stream { try? await stream.stopCapture() }
        if micEnabled { engine.stop(); engine.inputNode.removeTap(onBus:0) }
        audioQueue.sync {}
        let duration = elapsed()
        stateLock.withLock { active = false }; stream = nil; converters.removeAll()
        emit(["type":"stopped","time":duration])
    }
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid,
              let description = sampleBuffer.formatDescription else { return }
        let format = AVAudioFormat(cmAudioFormatDescription:description)
        var needed = 0
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut:&needed,
            bufferListOut:nil, bufferListSize:0, blockBufferAllocator:nil, blockBufferMemoryAllocator:nil,
            flags:0, blockBufferOut:nil)
        guard needed >= MemoryLayout<AudioBufferList>.size else { return }
        let storage = UnsafeMutableRawPointer.allocate(byteCount:needed, alignment:MemoryLayout<AudioBufferList>.alignment)
        defer { storage.deallocate() }
        let list = storage.bindMemory(to:AudioBufferList.self,capacity:1)
        var retained: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer,
            bufferListSizeNeededOut:nil, bufferListOut:list, bufferListSize:needed,
            blockBufferAllocator:nil, blockBufferMemoryAllocator:nil, flags:0, blockBufferOut:&retained)
        guard status == noErr else { emit(["type":"error","error":"Mac audio buffer could not be read."]); return }
        guard let pcm = AVAudioPCMBuffer(pcmFormat:format,bufferListNoCopy:list) else { return }
        withExtendedLifetime(retained) {
            packet(pcm,source:"system",timestamp:CMTimeGetSeconds(sampleBuffer.presentationTimeStamp))
        }
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        emit(["type":"error","error":"Mac audio capture stopped: \(error.localizedDescription)"])
    }
}

if CommandLine.arguments.contains("--check") {
    // Metadata-only diagnostic: no engine, devices, capture, or permission calls.
    let info = Bundle.main.infoDictionary ?? [:]
    let value: [String: Any] = ["type":"helper_check", "capture_started":false,
        "bundle_identifier":Bundle.main.bundleIdentifier ?? "",
        "microphone_usage":info["NSMicrophoneUsageDescription"] as? String ?? "",
        "system_audio_usage":info["NSScreenCaptureUsageDescription"] as? String ?? ""]
    let data = try! JSONSerialization.data(withJSONObject:value)
    FileHandle.standardOutput.write(data + Data([10]))
    exit(0)
}
let capture = MeetingCapture()
Task {
    while let line = readLine() {
        do {
            guard let data = line.data(using:.utf8),
                  let command = try JSONSerialization.jsonObject(with:data) as? [String:Any],
                  let action = command["type"] as? String else { continue }
            switch action {
            case "start": try await capture.start(microphone:command["microphone"] as? Bool ?? false,
                                                  system:command["system"] as? Bool ?? false)
            case "pause": try await capture.pause()
            case "resume": try await capture.resume()
            case "stop": await capture.stop(); capture.drainOutput(); exit(0)
            default: capture.emit(["type":"error","error":"Unknown recording action."])
            }
        } catch { capture.emit(["type":"error","error":error.localizedDescription]) }
    }
    await capture.stop(); capture.drainOutput(); exit(0)
}
RunLoop.main.run()

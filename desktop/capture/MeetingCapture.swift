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
    private var converters: [String: CaptureAudioConverter] = [:]
    private var startHost = 0.0
    private var pausedAt = 0.0
    private var pausedSeconds = 0.0
    private var active = false
    private var paused = false
    private var micEnabled = false
    private var timer: DispatchSourceTimer?
    private var processingJobID: String?
    private var processingHelperHash = ""
    // audioQueue owns pending output and the one-time processing announcement.
    private var processingAnnounced = false
    private var pendingSystemChunks: [ConvertedAudioChunk] = []
    private var microphoneFormat: AVAudioFormat?
    private var failed = false
    private let pcmFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
        sampleRate: 16000, channels: 1, interleaved: false)!

    private let systemFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
        sampleRate: 16000, channels: 2, interleaved: true)!

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
    private func packet(_ input: AVAudioPCMBuffer, source: String, timestamp: Double, observation: AppleCaptureObservation? = nil) {
        // The serial audio queue owns both converters; state snapshots use a lock.
        let timing = stateLock.withLock { (active, paused, startHost, pausedSeconds) }
        guard timing.0 && !timing.1 && !stateLock.withLock({failed}) else { return }
        let targetFormat = source == "system" ? systemFormat : pcmFormat
        let position = max(0, timestamp-timing.2-timing.3)
        do {
            var retained = input
            if source == "microphone", let jobID=processingJobID {
                guard let observation else {throw NSError(domain:"AppleCaptureRoute",code:9)}
                // Drain old accepted callbacks with their own capture-time state.
                // A stopped graph must not relabel or discard already copied PCM.
                let receipt=try observation.packet(jobID:jobID,helperHash:processingHelperHash,callback:input.format)
                if let previous=microphoneFormat, !captureFormatMatches(previous,input.format) {
                    throw NSError(domain:"AppleCaptureRoute",code:4,userInfo:[NSLocalizedDescriptionKey:"Microphone capture epoch changed; captured audio is preserved."])
                }
                retained=try appleMicrophoneCopy(input) // Validate every callback before any PCM advances.
                if !processingAnnounced {
                    microphoneFormat=input.format;stateLock.withLock {processingAnnounced=true}
                    emit(receipt)
                    emit(["type":"recording","microphone":true,"system":true])
                    try emitConverted(pendingSystemChunks,source:"system")
                    pendingSystemChunks.removeAll()
                }
            }
            var converter = converters[source]
            if let previous = converter,
               previous.inputFormat != retained.format || previous.discontinuous(at: position) {
                if processingJobID != nil { throw NSError(domain:"AppleCaptureRoute",code:5,
                    userInfo:[NSLocalizedDescriptionKey:"Capture format/timeline changed; restart recording. Audio is preserved."]) }
                try emitConverted(previous.finish(), source: source)
                emit(["type":"format_changed","source":source,"time":position])
                converters.removeValue(forKey: source)
                converter = nil
            }
            if converter == nil {
                converter = try CaptureAudioConverter(input: retained.format, output: targetFormat)
                converters[source] = converter
            }
            try emitConverted(converter!.convert(retained, at: position), source: source)
        } catch { failCapture(error) }
    }
    private func failCapture(_ error:Error) {
        let first=stateLock.withLock { () -> Bool in
            if failed { return false };failed=true;return true
        }
        guard first else{return}
        emit(["type":"error","error":error.localizedDescription])
        Task.detached {await self.stop(emitStopped:false);self.drainOutput();exit(1)}
    }
    private func emitConverted(_ chunks: [ConvertedAudioChunk], source: String) throws {
        if source == "system", processingJobID != nil, !processingAnnounced {
            guard pendingSystemChunks.count+chunks.count <= 64 else {throw NSError(domain:"AppleCaptureRoute",code:6)}
            pendingSystemChunks.append(contentsOf:chunks);return
        }
        for chunk in chunks {
            guard let pointer = chunk.pcm.floatChannelData?[0] else {
                throw NSError(domain:"MeetingCapture",code:6)
            }
            let channels = Int(chunk.pcm.format.channelCount)
            let audio = Data(bytes:pointer,count:Int(chunk.pcm.frameLength)*channels*4)
            emit(["type":"audio","source":source,"time":chunk.time,"rate":16000,
                  "channels":channels,"pcm":audio.base64EncodedString()])
        }
    }
    private func finishConversion() {
        // The serial queue drains copied input before EOF releases SRC lookahead.
        audioQueue.sync {
            for (source, converter) in converters {
                do { try emitConverted(converter.finish(), source:source) }
                catch { emit(["type":"error","error":error.localizedDescription]) }
            }
            converters.removeAll()
        }
    }

    func start(microphone: Bool, system: Bool, processingJob: String? = nil) async throws {
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
            config.sampleRate = 16000; config.channelCount = 2
            let capture = SCStream(filter: filter, configuration: config, delegate: self)
            try capture.addStreamOutput(self, type:.audio, sampleHandlerQueue: audioQueue)
            stream = capture
        }
        processingJobID = processingJob
        if processingJob != nil {
            guard microphone && system, processingJob!.range(of:"^[0-9a-f]{32}$",options:.regularExpression) != nil else {
                throw NSError(domain:"AppleCaptureRoute",code:7)
            }
            processingHelperHash = try ownedCaptureHash()
        }
        micEnabled = microphone
        stateLock.withLock { startHost = hostSeconds(); pausedSeconds = 0; paused = false; active = true }
        // Hold copied callbacks until engine.start returns. The real-time tap
        // never waits; the serial queue resumes on both success and failure.
        var audioSuspended = false
        if processingJobID != nil {audioQueue.suspend();audioSuspended=true}
        do {
            if let stream { try await stream.startCapture() }
            if microphone {
                let node = engine.inputNode
                if processingJobID != nil {
                    try node.setVoiceProcessingEnabled(true)
                    node.isVoiceProcessingInputMuted=false
                    node.isVoiceProcessingAGCEnabled=false
                    node.isVoiceProcessingBypassed=false
                    node.voiceProcessingOtherAudioDuckingConfiguration =
                        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking:false,duckingLevel:.min)
                }
                let format = node.outputFormat(forBus:0)
                guard format.sampleRate > 0 else { throw NSError(domain:"MeetingCapture",code:5,
                    userInfo:[NSLocalizedDescriptionKey:"No microphone is available."]) }
                node.installTap(onBus:0, bufferSize:2048, format:processingJobID == nil ? format : nil) { [weak self] buffer, time in
                    guard let self else { return }
                    // The tap owns the buffer only for this callback; copy before asynchronous conversion.
                    guard let copy = AVAudioPCMBuffer(pcmFormat:buffer.format, frameCapacity:buffer.frameLength) else { return }
                    copy.frameLength = buffer.frameLength
                    let src = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
                    let dst = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
                    for i in 0..<src.count { memcpy(dst[i].mData!, src[i].mData!, Int(src[i].mDataByteSize)) }
                    let observation=self.processingJobID == nil ? nil : observeAppleCapture(self.engine)
                    let timestamp = time.isHostTimeValid ? AVAudioTime.seconds(forHostTime:time.hostTime) : self.hostSeconds()
                    self.audioQueue.async { self.packet(copy,source:"microphone",timestamp:timestamp,observation:observation) }
                }
                engine.prepare(); try engine.start()
                if processingJobID != nil {
                    emit(["type":"capture_runtime","state_phase":"after_start",
                          "input_output_format":captureFormat(engine.inputNode.outputFormat(forBus:0)),
                          "output_input_format":captureFormat(engine.outputNode.inputFormat(forBus:0)),
                          "input_enabled":engine.inputNode.isVoiceProcessingEnabled,
                          "output_enabled":engine.outputNode.isVoiceProcessingEnabled,
                          "bypassed":engine.inputNode.isVoiceProcessingBypassed,"engine_running":engine.isRunning])
                }
            }
            if audioSuspended {audioQueue.resume();audioSuspended=false}
            let ticker = DispatchSource.makeTimerSource(queue:output)
            ticker.schedule(deadline:.now(), repeating:.milliseconds(250))
            ticker.setEventHandler { [weak self] in
                guard let self, self.stateLock.withLock({ self.active && (self.processingJobID == nil || self.processingAnnounced) }) else { return }
                self.emit(["type":"clock","time":self.elapsed(),"paused":self.stateLock.withLock({ self.paused })])
            }
            timer = ticker; ticker.resume()
            if processingJobID == nil {emit(["type":"recording","microphone":microphone,"system":system])}
        } catch {
            if audioSuspended {audioQueue.resume();audioSuspended=false}
            // Failed startup must emit an error, without a successful stop acknowledgement.
            await stop(emitStopped: false); throw error
        }
    }
    func pause() async throws {
        let timing = stateLock.withLock { (active, paused, startHost, pausedSeconds) }
        guard timing.0 && !timing.1 else { return }
        if processingJobID != nil {
            // The qualified app boundary ends the epoch after draining.
            if let stream {try await stream.stopCapture()}
            if micEnabled {engine.pause()}
            finishConversion()
            stateLock.withLock {pausedAt=hostSeconds();paused=true}
            emit(["type":"paused","time":elapsed()]);return
        }
        if let stream { try await stream.stopCapture() }
        if micEnabled { engine.pause() }
        // Drain copied audio before changing timeline state.
        finishConversion()
        stateLock.withLock { pausedAt = hostSeconds(); paused = true }
        emit(["type":"paused","time":elapsed()])
    }
    func resume() async throws {
        guard stateLock.withLock({ active && paused }) else { return }
        guard processingJobID == nil else {throw NSError(domain:"AppleCaptureRoute",code:8,
            userInfo:[NSLocalizedDescriptionKey:"Restart the recording after a processed microphone Pause."])}
        stateLock.withLock { pausedSeconds += hostSeconds()-pausedAt; paused = false }
        if let stream { try await stream.startCapture() }
        if micEnabled { try engine.start() }
        emit(["type":"recording","time":elapsed()])
    }
    func stop(emitStopped: Bool = true) async {
        guard stateLock.withLock({ active }) else { return }
        timer?.cancel(); timer = nil
        if let stream { try? await stream.stopCapture() }
        if micEnabled { engine.stop(); engine.inputNode.removeTap(onBus:0) }
        finishConversion()
        if processingJobID != nil {try? engine.inputNode.setVoiceProcessingEnabled(false);engine.reset()}
        let duration = elapsed()
        stateLock.withLock { active = false }; stream = nil; converters.removeAll()
        if emitStopped { emit(["type":"stopped","time":duration]) }
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

@main
struct CaptureMain {
    static func main() {

        if CommandLine.arguments.contains("--check") {
            // Metadata-only diagnostic: no engine, devices, capture, or permission calls.
            let info = Bundle.main.infoDictionary ?? [:]
            let value: [String: Any] = ["type":"helper_check", "capture_started":false,
                "bundle_identifier":Bundle.main.bundleIdentifier ?? "",
                "microphone_usage":info["NSMicrophoneUsageDescription"] as? String ?? "",
                "system_audio_usage":info["NSScreenCaptureUsageDescription"] as? String ?? "",
                "capability_version":appleCaptureCapability,"export_contract":appleCaptureExport]
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
                                                          system:command["system"] as? Bool ?? false,
                        processingJob:command["capture_export_contract"] as? String == appleCaptureExport ? command["job_id"] as? String : nil)
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

    }
}

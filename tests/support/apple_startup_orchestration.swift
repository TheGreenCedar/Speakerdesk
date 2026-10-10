import Foundation
import AVFoundation
final class StartupQueuePeer {
    var processingJobID:String?="fixture"
    var processingAnnounced=false
    var pendingSystemChunks:[ConvertedAudioChunk]=[]
    var events:[[String:Any]]=[]
    func emit(_ value:[String:Any]) {events.append(value)}
    private func emitConverted(_ chunks: [ConvertedAudioChunk], source: String) throws {
        if source == "system", processingJobID != nil, !processingAnnounced {
            guard pendingSystemChunks.count+chunks.count <= 64 else {
                throw NSError(domain:"AppleCaptureRoute",code:6,userInfo:[NSLocalizedDescriptionKey:
                    "Mac audio reached the startup buffer limit before the processed microphone was ready."])
            }
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
    func system(_ chunks:[ConvertedAudioChunk]) throws {try emitConverted(chunks,source:"system")}
    func acceptedMicrophone(input:AVAudioPCMBuffer,observation:AppleCaptureObservation) throws {
        let receipt=try observation.packet(jobID:String(repeating:"a",count:32),helperHash:String(repeating:"b",count:64),callback:input.format)
        _=try appleMicrophoneCopy(input)
        processingAnnounced=true
        emit(receipt)
        try emitConverted(pendingSystemChunks,source:"system")
        pendingSystemChunks.removeAll()
    }
}

final class StartupScenario {
    let lock=NSLock()
    let peer=StartupQueuePeer()
    var stopped=false
    var starts=0
    var cleanup=0
    var recording=false
    func state() -> AppleMicrophoneStartupState {
        lock.withLock {stopped ? .stopped : peer.processingAnnounced ? .ready : .waiting}
    }
    func accept(_ input:AVAudioPCMBuffer,_ observation:AppleCaptureObservation) throws {
        try lock.withLock {try peer.acceptedMicrophone(input:input,observation:observation)}
    }
    func end() {lock.withLock {stopped=true;cleanup+=1}}
    func system(_ chunks:[ConvertedAudioChunk],fail:Bool=false) throws {
        try lock.withLock {
            starts+=1
            precondition(peer.processingAnnounced)
            if fail {throw NSError(domain:"SyntheticSystem",code:99)}
            for chunk in chunks {try peer.system([chunk])}
        }
    }
    func announce() {lock.withLock {recording=true;peer.emit(["type":"recording"])}}
}
@main struct StartupChecks {
    static func main() async throws {
        let format=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:16000,channels:2,interleaved:true)!
        let converter=try CaptureAudioConverter(input:format,output:format)
        var chunks:[ConvertedAudioChunk]=[]
        for i in 0..<65 {
            let pcm=AVAudioPCMBuffer(pcmFormat:format,frameCapacity:512)!;pcm.frameLength=512
            for j in 0..<1024 {pcm.floatChannelData![0][j]=Float(i+1)/1000}
            chunks += try converter.convert(pcm,at:Double(i)*512/16000)
        }
        precondition(chunks.count==65)
        let micFormat=AVAudioFormat(commonFormat:.pcmFormatFloat32,sampleRate:48000,channels:1,interleaved:false)!
        let mic=AVAudioPCMBuffer(pcmFormat:micFormat,frameCapacity:2048)!;mic.frameLength=2048
        for j in 0..<2048 {mic.floatChannelData![0][j]=0}
        let readbacks=["input_enabled":true,"output_enabled":true,"engine_running":true,"bypassed":false,"manual_rendering":false,"input_muted":false,"agc_enabled":false]
        let observation=AppleCaptureObservation(input:micFormat,output:micFormat,readbacks:readbacks)
        var checks:[String:Bool]=[:]
        let old=StartupQueuePeer();var oldError=0
        do {for chunk in chunks {try old.system([chunk])};try old.acceptedMicrophone(input:mic,observation:observation)}
        catch {oldError=(error as NSError).code}
        precondition(oldError==6 && old.events.isEmpty && old.pendingSystemChunks.count==64)
        checks["old_system_first_reproduces_error6"]=true
        let late=StartupScenario()
        let arrival=Task {try await Task.sleep(nanoseconds:60_000_000);try late.accept(mic,observation)}
        try await startSystemAfterAppleMicrophone(timeout:1,readiness:{late.state()},startSystem:{try late.system(chunks)})
        try await arrival.value;late.announce()
        precondition(late.starts==1 && late.recording && late.peer.events.count==67)
        precondition(late.peer.events.first?["type"] as? String=="capture_processing")
        precondition(late.peer.events.last?["type"] as? String=="recording")
        for (packet,chunk) in zip(late.peer.events.dropFirst().dropLast(),chunks) {
            let data=Data(bytes:chunk.pcm.floatChannelData![0],count:Int(chunk.pcm.frameLength)*8)
            precondition(packet["time"] as? Double==chunk.time && packet["pcm"] as? String==data.base64EncodedString())
        }
        checks["late_mic_starts_system_once_after_valid_receipt"]=true
        checks["65_system_packets_keep_exact_pcm_and_timestamps"]=true
        let asynchronous=StartupScenario();try asynchronous.accept(mic,observation)
        try await startSystemAfterAppleMicrophone(readiness:{asynchronous.state()},startSystem:{try await Task.sleep(nanoseconds:1_000_000);try asynchronous.system(chunks)})
        asynchronous.announce();precondition(asynchronous.starts==1 && asynchronous.peer.events.count==67)
        checks["asynchronous_system_start_after_receipt"]=true
        let missing=StartupScenario();var missingCode=0
        do {try await startSystemAfterAppleMicrophone(timeout:0.04,readiness:{missing.state()},startSystem:{try missing.system(chunks)});missing.announce()}
        catch {missingCode=(error as NSError).code;missing.end()}
        precondition(missingCode==10 && missing.starts==0 && !missing.recording && missing.cleanup==1)
        checks["missing_mic_timeout_does_not_start_system_or_recording"]=true
        let stopped=StartupScenario();stopped.end();var stoppedCode=0
        do {try await startSystemAfterAppleMicrophone(readiness:{stopped.state()},startSystem:{try stopped.system(chunks)})}
        catch {stoppedCode=(error as NSError).code}
        precondition(stoppedCode==11 && stopped.starts==0 && !stopped.recording)
        checks["stopped_graph_refuses_system_start"]=true
        let cancelled=StartupScenario()
        let wait=Task {try await startSystemAfterAppleMicrophone(timeout:1,readiness:{cancelled.state()},startSystem:{try cancelled.system(chunks)})}
        try await Task.sleep(nanoseconds:10_000_000);wait.cancel();var cancellation=false
        do {try await wait.value;cancelled.announce()} catch is CancellationError {cancellation=true;cancelled.end()}
        precondition(cancellation && cancelled.starts==0 && !cancelled.recording && cancelled.cleanup==1)
        checks["cancelled_wait_never_starts_system"]=true
        let invalid=StartupScenario();var invalidReadbacks=readbacks;invalidReadbacks["engine_running"]=false;var invalidCode=0
        do {try invalid.accept(mic,AppleCaptureObservation(input:micFormat,output:micFormat,readbacks:invalidReadbacks))}
        catch {invalidCode=(error as NSError).code;invalid.end()}
        do {try await startSystemAfterAppleMicrophone(readiness:{invalid.state()},startSystem:{try invalid.system(chunks)})} catch {}
        precondition(invalidCode==2 && invalid.starts==0 && invalid.peer.events.isEmpty && !invalid.recording)
        checks["invalid_first_tap_cannot_qualify_or_start_system"]=true
        let failed=StartupScenario();try failed.accept(mic,observation);var systemCode=0
        do {try await startSystemAfterAppleMicrophone(readiness:{failed.state()},startSystem:{try failed.system(chunks,fail:true)});failed.announce()}
        catch {systemCode=(error as NSError).code;failed.end()}
        precondition(systemCode==99 && failed.starts==1 && !failed.recording && failed.cleanup==1)
        checks["system_start_failure_after_receipt_has_no_recording_ack"]=true
        let duringSystem=StartupScenario();try duringSystem.accept(mic,observation);var duringCode=0
        do {
            try await startSystemAfterAppleMicrophone(readiness:{duringSystem.state()},startSystem:{
                try duringSystem.system([])
                try await Task.sleep(nanoseconds:1_000_000)
                duringSystem.end()
            })
            duringSystem.announce()
        } catch {duringCode=(error as NSError).code}
        precondition(duringCode==11 && duringSystem.starts==1 && !duringSystem.recording)
        checks["mic_failure_during_system_start_refuses_recording"]=true
        let afterCancel=StartupScenario();try afterCancel.accept(mic,observation)
        let cancelInSystem=Task {
            try await startSystemAfterAppleMicrophone(readiness:{afterCancel.state()},startSystem:{
                try afterCancel.system([])
                try? await Task.sleep(nanoseconds:100_000_000)
            })
            afterCancel.announce()
        }
        try await Task.sleep(nanoseconds:10_000_000);cancelInSystem.cancel();var afterCancelled=false
        do {try await cancelInSystem.value} catch is CancellationError {afterCancelled=true}
        precondition(afterCancelled && !afterCancel.recording)
        checks["cancellation_after_system_await_refuses_recording"]=true
        let stopOwner=AppleCaptureStopOwner()
        let stopState=StartupScenario()
        let cleanup:() async -> Void = {
            stopState.lock.withLock {stopState.cleanup+=1}
            try? await Task.sleep(nanoseconds:40_000_000)
            stopState.lock.withLock {stopState.stopped=true}
        }
        async let failureStop:Void=stopOwner.run(cleanup)
        async let startupCatchStop:Void=stopOwner.run(cleanup)
        _=await(failureStop,startupCatchStop)
        precondition(stopState.cleanup==1 && stopState.stopped)
        checks["actual_stop_owner_runs_once_and_both_failure_paths_join"]=true
        await stopOwner.reset()
        await stopOwner.run(cleanup)
        precondition(stopState.cleanup==2)
        checks["completed_stop_owner_resets_for_next_start"]=true
        let result:[String:Any]=["checks":checks,"count":checks.count,"all_passed":true,"old_error_code":oldError,"old_emitted_packets":old.events.count,"generated_stereo_frames":65*512,"device_opened":false,"capture_started":false]
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]))
    }
}

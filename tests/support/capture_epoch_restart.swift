import Foundation

typealias SCStream = NSObject
typealias AVAudioPCMBuffer = Int
typealias AppleCaptureObservation = Int

final class EpochPeer {
    let stateLock=NSLock()
    let audioQueue=DispatchQueue(label:"speakerdesk.epoch.test")
    var converters:[String:Int]=[:]
    var pendingSystemChunks:[Int]=[]
    var microphoneFormat:Int?
    var microphoneObservation:Int?
    var microphoneDevice:UInt32?
    var resumeBoundary:AppleMicrophoneResumeBoundary?
    var captureHostFloor=0.0
    var captureGeneration:UInt64=0
    var processingAnnounced=false
    var failed=false
    var startHost=0.0
    var pausedAt=0.0
    var pausedSeconds=0.0
    var paused=false
    var active=false
    var startupPhase="idle"
    var processingJobID:String?
    var processingHelperHash=""
    var micEnabled=false
    var stream:SCStream?
    var errorEvents:[[String:Any]]=[]
    func emit(_ value:[String:Any]) {errorEvents.append(value)}
    var emittedReceipts:[String]=[]
    var emittedPCMJobs:[String]=[]
    private func resetCaptureEpoch() {
        // stopOwner has joined old cleanup. Drain any already queued old work
        // while inactive before opening a new job/clock/processing identity.
        audioQueue.sync {
            converters.removeAll()
            pendingSystemChunks.removeAll()
            microphoneFormat=nil
            microphoneObservation=nil;microphoneDevice=nil
            stateLock.withLock {
                captureGeneration += 1
                processingAnnounced=false
                failed=false
                startHost=0;pausedAt=0;pausedSeconds=0;paused=false
                resumeBoundary=nil;captureHostFloor=0
                startupPhase="idle"
            }
        }
        processingJobID=nil;processingHelperHash="";micEnabled=false
        stateLock.withLock {stream=nil}
    }
    private func acceptedSystemGeneration(_ origin:SCStream) -> UInt64? {
        stateLock.withLock {
            guard active && !paused && !failed && stream === origin else {return nil}
            return captureGeneration
        }
    }
    private func packet(_ input: AVAudioPCMBuffer, source: String, timestamp: Double, generation: UInt64, observation: AppleCaptureObservation? = nil) {
        // The serial audio queue owns both converters; state snapshots use a lock.
        let timing = stateLock.withLock { (active, paused, captureGeneration, captureHostFloor, resumeBoundary) }
        guard timing.0 && timing.2 == generation && AppleMicrophoneResumeBoundary.accepts(timestamp,after:timing.3) &&
            (!timing.1 || (source == "microphone" && timing.4 != nil)) && !stateLock.withLock({failed}) else { return }

        // Synthetic publication peer follows the unchanged first-receipt gate.
        // The production entry guard above decides whether this job may see PCM.
        if !processingAnnounced {emittedReceipts.append(processingJobID!);processingAnnounced=true}
        emittedPCMJobs.append(processingJobID!)
    }
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        guard acceptedSystemGeneration(stream) != nil else {return}
        emit(["type":"error","error":"Mac audio capture stopped: \(error.localizedDescription)"])
    }
    func reset() {resetCaptureEpoch()}
    func systemGeneration(_ origin:SCStream) ->UInt64? {acceptedSystemGeneration(origin)}
    func mic(_ generation:UInt64) {packet(0,source:"microphone",timestamp:0,generation:generation)}
}

@main struct CheckEpoch {
    static func main() throws {
        let peer=EpochPeer()
        peer.captureGeneration=7;peer.processingAnnounced=true;peer.failed=true
        peer.microphoneFormat=9;peer.pendingSystemChunks=[11,12];peer.converters=["microphone":55]
        peer.startHost=123;peer.pausedAt=9;peer.pausedSeconds=8;peer.paused=true
        peer.startupPhase="system_start";peer.processingJobID="old";peer.processingHelperHash="oldhash";peer.micEnabled=true
        let oldStream=SCStream();peer.stream=oldStream
        var checks:[String:Bool]=[:]
        peer.reset()
        precondition(peer.captureGeneration==8 && !peer.failed && !peer.processingAnnounced)
        precondition(peer.microphoneFormat==nil && peer.pendingSystemChunks.isEmpty && peer.converters.isEmpty)
        checks["failed_and_announced_and_format_buffers_reset"]=true
        precondition(peer.startHost==0 && peer.pausedAt==0 && peer.pausedSeconds==0 && !peer.paused)
        precondition(peer.processingJobID==nil && peer.processingHelperHash.isEmpty && !peer.micEnabled && peer.stream==nil)
        checks["job_clock_and_producer_state_reset"]=true
        peer.processingJobID="new";peer.active=true
        let newStream=SCStream();peer.stream=newStream
        peer.mic(7)
        precondition(peer.emittedReceipts.isEmpty && peer.emittedPCMJobs.isEmpty)
        checks["late_old_microphone_generation_rejected"]=true
        precondition(peer.systemGeneration(oldStream)==nil && peer.systemGeneration(newStream)==8)
        checks["old_system_stream_identity_rejected"]=true
        let callbackError=NSError(domain:"SyntheticSCStream",code:77)
        peer.stream(oldStream,didStopWithError:callbackError)
        precondition(peer.errorEvents.isEmpty)
        peer.stream(newStream,didStopWithError:callbackError)
        precondition(peer.errorEvents.count==1 && peer.errorEvents[0]["type"] as? String=="error")
        checks["old_stream_error_ignored_and_current_stream_error_retained"]=true
        peer.mic(8);peer.mic(8)
        precondition(peer.emittedReceipts==["new"] && peer.emittedPCMJobs==["new","new"])
        checks["new_generation_emits_new_job_receipt_once"]=true
        peer.active=false;peer.reset();peer.processingJobID="third";peer.active=true
        peer.mic(8);peer.mic(9)
        precondition(peer.emittedReceipts==["new","third"] && peer.emittedPCMJobs.last=="third")
        checks["successful_epoch_restart_requires_new_receipt"]=true
        peer.failed=true;peer.active=false;peer.reset();peer.processingJobID="after-failure";peer.active=true;peer.mic(10)
        precondition(peer.emittedReceipts.last=="after-failure" && peer.captureGeneration==10)
        checks["failed_epoch_restart_accepts_current_callback"]=true
        peer.failed=true
        let current=SCStream();peer.stream=current
        precondition(peer.systemGeneration(current)==nil)
        peer.active=false;peer.failed=false
        precondition(peer.systemGeneration(current)==nil)
        checks["failed_or_inactive_stream_refused"]=true
        let data=try JSONSerialization.data(withJSONObject:["checks":checks,"count":checks.count,"all_passed":true,"device_opened":false,"capture_started":false],options:[.prettyPrinted,.sortedKeys])
        FileHandle.standardOutput.write(data)
    }
}

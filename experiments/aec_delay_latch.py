"""Experimental native-delay stability policy. No product caller or ASR gate."""
import math


class DelayLatch:
    """Three distinct recent clock matches; epoch reset and bounded uncertainty.

    Delays are integer milliseconds already rounded by EchoMixer. ABI3 then
    floors to64-sample/4ms blocks. Adjacent blocks are one stability group;
    retaining an established member avoids boundary jitter. A larger route
    transition returns unknown until three new observations support its group.
    Unknown means preserve raw mic in the experiment, not a non-speech claim.
    """
    def __init__(self):
        self.reset(None)

    def reset(self, epoch):
        self.epoch = epoch; self.last_observation = None
        self.recent = []; self.held = None; self.started = None
        self.reason = 'reset'

    def observe(self, delay_ms, center, observed_sample, epoch):
        if epoch is not self.epoch:
            self.reset(epoch)
        if (type(delay_ms) is not int or not 0 <= delay_ms <= 1000
                or type(center) not in (int, float) or not math.isfinite(center)
                or type(observed_sample) is not int or center > observed_sample):
            self.recent = []; self.held = None; self.started = None
            self.reason = 'unknown_observation'
            return -1
        if self.last_observation is not None and center <= self.last_observation:
            # Stale/duplicate observations never accrue confirmation. The same
            # matched delay may still be used while its evidence is recent.
            if observed_sample-center > 16000:
                self.recent = []; self.held = None; self.started = None
                self.reason = 'expired_observation'
                return -1
            self.reason = 'duplicate_or_rejected_observation'
            return self.held*4 if self.held is not None and self.started is None else -1
        self.last_observation = center
        block = delay_ms//4
        if self.held is not None and abs(block-self.held) <= 1:
            self.recent = []; self.started = None
            self.reason = 'held_boundary_group'
            return self.held*4
        if self.started is None:
            self.started = observed_sample
        self.recent.append((center, block)); self.recent = self.recent[-3:]
        if len(self.recent) == 3:
            values = sorted(value for _, value in self.recent)
            if values[-1]-values[0] <= 1:
                self.held = values[1]; self.recent = []; self.started = None
                self.reason = 'accepted_three_distinct_matches'
                return self.held*4
        if observed_sample-self.started > 16000:
            self.held = None; self.recent = []
            # Preserve the unresolved transition start rather than repeatedly
            # restarting its age and secretly extending stale-path validity.
            self.reason = 'unresolved_route_expired'
        else:
            self.reason = 'awaiting_stable_matches'
        return -1

    def evidence(self):
        return dict(reason=self.reason, held_block=self.held,
                    pending_observations=[list(pair) for pair in self.recent],
                    last_observation=self.last_observation, transition_started=self.started)

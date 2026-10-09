// Explicit-reference capture DSP. Called after host-clock alignment.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) int speakerdesk_echo_abi(void);

// One owner thread; 16 kHz Float32, exactly 160 sample frames (10 ms).
// Render: interleaved stereo (320 floats). Microphone/output: mono (160 floats).
// The caller aligns both streams by capture host timestamps before this call.
// delay_ms is a validated external reference/capture lag, or -1 while unknown.
// `render` is a reference only: this library never opens or plays audio.
__attribute__((visibility("default"))) void* speakerdesk_echo_create(void);
__attribute__((visibility("default"))) void speakerdesk_echo_destroy(void* state);
__attribute__((visibility("default"))) int speakerdesk_echo_reset(void* state);
__attribute__((visibility("default"))) int speakerdesk_echo_process(void* state, const float* render,
                            const float* microphone, float* clean_microphone,
                            size_t samples, int delay_ms);

#ifdef __cplusplus
}
#endif

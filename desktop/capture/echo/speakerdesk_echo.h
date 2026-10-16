// Explicit-reference capture DSP. Called after host-clock alignment.
#pragma once
#include <stddef.h>

#ifdef _WIN32
#define SPEAKERDESK_ECHO_EXPORT __declspec(dllexport)
#else
#define SPEAKERDESK_ECHO_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

SPEAKERDESK_ECHO_EXPORT int speakerdesk_echo_abi(void);

// One owner thread; 16 kHz Float32, exactly 160 sample frames (10 ms).
// Render: interleaved stereo (320 floats). Microphone/output: mono (160 floats).
// The caller aligns both streams by capture host timestamps before this call.
// delay_ms is a validated external reference/capture lag, or -1 while unknown.
// `render` is a reference only: this library never opens or plays audio.
// ABI 5 output: linear echo cancellation and default 12 dB noise suppression,
// without AGC. Fixed latency is 160 samples (10 ms); the caller must discard
// that prefix and supply enough zero frames to drain it on Pause/Stop.
SPEAKERDESK_ECHO_EXPORT void* speakerdesk_echo_create(void);
SPEAKERDESK_ECHO_EXPORT void speakerdesk_echo_destroy(void* state);
SPEAKERDESK_ECHO_EXPORT int speakerdesk_echo_reset(void* state);
SPEAKERDESK_ECHO_EXPORT int speakerdesk_echo_process(void* state, const float* render,
                            const float* microphone, float* clean_microphone,
                            size_t samples, int delay_ms);

#ifdef __cplusplus
}
#endif

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
// ABI 6: linear prediction error only, fixed 64-sample latency. The same state
// owns the default LocalVQE v1.2 CPU model (hard gate and AGC off). Required
// model_path_utf8 is the absolute bundled GGUF; no download or path fallback.
// Remove the linear prefix before framing enhance(), while retaining the same
// original-coordinate mono projection of the corrected render. On Pause/Stop,
// supply zero mic AND render beyond EOF and feed continued linear drain through
// enhance() before resetting BOTH histories. Never trim between stages.
SPEAKERDESK_ECHO_EXPORT void* speakerdesk_echo_create(const char* model_path_utf8);
SPEAKERDESK_ECHO_EXPORT void speakerdesk_echo_destroy(void* state);
SPEAKERDESK_ECHO_EXPORT int speakerdesk_echo_reset(void* state);
SPEAKERDESK_ECHO_EXPORT int speakerdesk_echo_process(void* state, const float* render,
                            const float* microphone, float* clean_microphone,
                            size_t samples, int delay_ms);

// Exactly 256 mono samples each. reference=(left+right)/2 from the SAME corrected
// render sent to process(), paired after removing its 64-sample prefix. Clamp
// the linear residual to [-1,1]; do not normalize output. Fixed learned delay
// is 256 samples: the composed coordinate delay is 320 samples (20 ms).
SPEAKERDESK_ECHO_EXPORT int speakerdesk_echo_enhance(void* state, const float* reference,
                            const float* linear, float* clean_microphone, size_t samples);

#ifdef __cplusplus
}
#endif

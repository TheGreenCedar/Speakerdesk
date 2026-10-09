// Explicit-reference CPU acoustic echo cancellation. No device playback or model inference.
#include "speakerdesk_echo.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "api/audio/audio_processing.h"
#include "api/environment/environment_factory.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"

namespace {
constexpr size_t kFrame = 160;
webrtc::EchoCanceller3Config Configuration(bool stereo) {
  webrtc::EchoCanceller3Config config;
  config.filter.export_linear_aec_output = true;
  config.multi_channel.detect_stereo_content = false;
  config.delay.use_external_delay_estimator = true;
  // Retain a 128 ms room tail. A shorter startup filter converges on the
  // direct speaker path first; upstream also shortens the stereo coarse filter.
  config.filter.refined.length_blocks = 32;
  config.filter.refined_initial.length_blocks = stereo ? 8 : 13;
  config.filter.coarse.length_blocks = 32;
  config.filter.initial_state_seconds = 1.f;
  config.filter.coarse_initial.length_blocks = stereo ? 8 : 13;
  config.filter.coarse.rate = .95f;
  config.filter.coarse_initial.rate = stereo ? 1.f : .95f;
  config.filter.refined_initial.leakage_converged = .05f;
  // Quiet normalized capture must still excite the linear adaptive filters.
  config.filter.refined.noise_gate = 200000.f;
  config.filter.refined_initial.noise_gate = 200000.f;
  config.filter.coarse.noise_gate = 200000.f;
  config.filter.coarse_initial.noise_gate = 200000.f;
  config.echo_removal_control.has_clock_drift = true;
  if (!webrtc::EchoCanceller3Config::Validate(&config))
    throw std::invalid_argument("invalid capture echo configuration");
  return config;
}
struct Bank {
  webrtc::Environment env;
  webrtc::EchoCanceller3Config config;
  webrtc::AudioBuffer render;
  webrtc::AudioBuffer capture{16000, 1, 16000, 1, 16000, 1};
  webrtc::AudioBuffer linear{16000, 1, 16000, 1, 16000, 1};
  std::unique_ptr<webrtc::EchoCanceller3> echo;
  int channels;
  Bank(const webrtc::Environment& environment, int render_channels)
      : env(environment), config(Configuration(render_channels == 2)),
        render(16000, render_channels, 16000, render_channels, 16000, render_channels),
        channels(render_channels) { Reset(); }
  void Reset() {
    // nullptr explicitly selects ordinary residual-echo DSP. Only the linear
    // prediction error is returned: no AGC, VAD, neural model or suppressor gain.
    echo = std::make_unique<webrtc::EchoCanceller3>(
        env, config, std::nullopt, nullptr, 16000, channels, 1);
  }
  void Process(const float* const* reference, const float* microphone,
               float* output, int delay_ms) {
    render.CopyFrom(reference, webrtc::StreamConfig(16000, channels));
    echo->AnalyzeRender(&render);
    if (delay_ms >= 0) echo->SetAudioBufferDelay(delay_ms);
    const float* near[] = {microphone};
    float* cleaned[] = {output};
    capture.CopyFrom(near, webrtc::StreamConfig(16000, 1));
    echo->AnalyzeCapture(&capture);
    echo->ProcessCapture(&capture, &linear, false);
    linear.CopyTo(webrtc::StreamConfig(16000, 1), cleaned);
  }
};
struct State {
  webrtc::Environment env = webrtc::CreateEnvironment();
  Bank mono{env, 1};
  Bank stereo{env, 2};
  float stereo_weight = 0.f;
  int stereo_hold = 0;
  void Reset() { mono.Reset();stereo.Reset();stereo_weight=0;stereo_hold=0; }
};
}
extern "C" int speakerdesk_echo_abi(void) { return 3; }
extern "C" void* speakerdesk_echo_create(void) {
  try { return new State; } catch (...) { return nullptr; }
}
extern "C" void speakerdesk_echo_destroy(void* state) { delete static_cast<State*>(state); }
extern "C" int speakerdesk_echo_reset(void* state) {
  if (!state) return -1;
  try { static_cast<State*>(state)->Reset();return 0; } catch (...) { return -2; }
}
extern "C" int speakerdesk_echo_process(void* state, const float* render,
                                        const float* microphone, float* cleaned,
                                        size_t samples, int delay_ms) {
  if (!state || !render || !microphone || !cleaned || samples != kFrame ||
      delay_ms < -1 || delay_ms > 1000) return -1;
  for (size_t i=0;i<samples;++i)
    if (!std::isfinite(microphone[i]) || std::abs(microphone[i]) > 1) return -1;
  for (size_t i=0;i<samples*2;++i)
    if (!std::isfinite(render[i]) || std::abs(render[i]) > 1) return -1;
  try {
    auto& s=*static_cast<State*>(state);
    float left[kFrame],right[kFrame],center[kFrame],one[kFrame],two[kFrame];
    float difference=0,power=0;
    for (size_t i=0;i<kFrame;++i) {
      left[i]=render[2*i];right[i]=render[2*i+1];center[i]=.5f*(left[i]+right[i]);
      difference+=(left[i]-right[i])*(left[i]-right[i]);
      power+=left[i]*left[i]+right[i]*right[i];
    }
    const float* mono[] = {center};const float* stereo[] = {left,right};
    s.mono.Process(mono,microphone,one,delay_ms);
    s.stereo.Process(stereo,microphone,two,delay_ms);
    // Both histories stay warm. Changing render dimensionality must not reset
    // framing or erase near-end samples when another app begins stereo playback.
    if (difference > std::max(1.e-8f,power*1.e-4f)) s.stereo_hold=300;
    else if(s.stereo_hold>0) --s.stereo_hold;
    const float target = s.stereo_hold>0 ? 1.f : 0.f;
    for (size_t i=0;i<kFrame;++i) {
      s.stereo_weight+=std::clamp(target-s.stereo_weight,-.01f,.01f);
      cleaned[i]=(1-s.stereo_weight)*one[i]+s.stereo_weight*two[i];
    }
    return 0;
  } catch (...) { return -2; }
}

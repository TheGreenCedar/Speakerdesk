cmake_minimum_required(VERSION 3.19)
project(SpeakerdeskEcho LANGUAGES C CXX)
if(APPLE)
  enable_language(OBJC)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0)
set(BUILD_TESTING OFF CACHE BOOL "No upstream test downloads" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "No upstream tests" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "Project-local build" FORCE)
set(ABSL_USE_EXTERNAL_GOOGLETEST OFF CACHE BOOL "No test dependencies" FORCE)
set(ABSL_FIND_GOOGLETEST OFF CACHE BOOL "No test dependencies" FORCE)
add_subdirectory("${WEBRTC_SOURCE}/third_party/abseil-cpp" abseil EXCLUDE_FROM_ALL)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/sources.json" source_json)
string(JSON source_count LENGTH "${source_json}")
math(EXPR source_last "${source_count}-1")
foreach(index RANGE ${source_last})
  string(JSON relative GET "${source_json}" ${index})
  list(APPEND echo_sources "${WEBRTC_SOURCE}/${relative}")
endforeach()
if(WIN32)
  list(FILTER echo_sources EXCLUDE REGEX "(default_task_queue_factory_gcd\\.cc|gcd_helpers\\.m|task_queue_gcd\\.cc)$")
  foreach(relative
      api/task_queue/default_task_queue_factory_win.cc
      rtc_base/task_queue_win.cc rtc_base/event.cc rtc_base/platform_thread.cc
      rtc_base/synchronization/yield_policy.cc
      common_audio/resampler/sinc_resampler_sse.cc
      common_audio/third_party/ooura/fft_size_128/ooura_fft_sse2.cc)
    list(APPEND echo_sources "${WEBRTC_SOURCE}/${relative}")
  endforeach()
  # Match upstream's runtime CPU dispatch; only these translation units require AVX2.
  foreach(relative
      common_audio/resampler/sinc_resampler_avx2.cc
      modules/audio_processing/aec3/adaptive_fir_filter_avx2.cc
      modules/audio_processing/aec3/adaptive_fir_filter_erl_avx2.cc
      modules/audio_processing/aec3/fft_data_avx2.cc
      modules/audio_processing/aec3/matched_filter_avx2.cc
      modules/audio_processing/aec3/vector_math_avx2.cc)
    list(APPEND echo_sources "${WEBRTC_SOURCE}/${relative}")
    set_source_files_properties("${WEBRTC_SOURCE}/${relative}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endforeach()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Upstream uses Clang vector indexing; MSVC exposes the same lanes as union fields.
    set(matched_source "${WEBRTC_SOURCE}/modules/audio_processing/aec3/matched_filter_avx2.cc")
    file(READ "${matched_source}" matched_filter)
    string(REPLACE "s_inst_hadd_256[" "s_inst_hadd_256.m256_f32[" matched_filter "${matched_filter}")
    string(REPLACE "e_128[" "e_128.m128_f32[" matched_filter "${matched_filter}")
    string(REPLACE "sum[" "sum.m128_f32[" matched_filter "${matched_filter}")
    set(matched_port "${CMAKE_CURRENT_BINARY_DIR}/matched_filter_avx2.cc")
    file(WRITE "${matched_port}" "${matched_filter}")
    list(REMOVE_ITEM echo_sources "${matched_source}")
    list(APPEND echo_sources "${matched_port}")
    set_source_files_properties("${matched_port}" PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated}/experiments")
add_custom_command(OUTPUT "${generated}/experiments/registered_field_trials.h"
  COMMAND "${Python3_EXECUTABLE}" "${WEBRTC_SOURCE}/experiments/field_trials.py" header
    --output "${generated}/experiments/registered_field_trials.h"
  DEPENDS "${WEBRTC_SOURCE}/experiments/field_trials.py")
add_library(speakerdesk_echo SHARED "${generated}/experiments/registered_field_trials.h" speakerdesk_echo.cc ${echo_sources})
target_include_directories(speakerdesk_echo PRIVATE "${WEBRTC_SOURCE}" "${generated}")
target_compile_definitions(speakerdesk_echo PRIVATE WEBRTC_APM_DEBUG_DUMP=0 NDEBUG)
target_link_libraries(speakerdesk_echo PRIVATE absl::strings absl::base absl::any_invocable)
if(WIN32)
  target_compile_definitions(speakerdesk_echo PRIVATE WEBRTC_WIN NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  target_compile_options(speakerdesk_echo PRIVATE /O2)
  target_link_libraries(speakerdesk_echo PRIVATE winmm)
else()
  target_compile_definitions(speakerdesk_echo PRIVATE WEBRTC_POSIX WEBRTC_MAC)
  target_compile_options(speakerdesk_echo PRIVATE -O2 -fintegrated-cc1 -Wno-nullability-completeness)
  target_link_libraries(speakerdesk_echo PRIVATE "-framework CoreServices")
endif()

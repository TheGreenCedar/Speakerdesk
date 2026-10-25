# Owned composition: no LocalVQE CMake hooks, alternate models, downloads or GPU.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
foreach(option GGML_NATIVE GGML_BACKEND_DL GGML_CPU_ALL_VARIANTS GGML_BUILD_TESTS
    GGML_BUILD_EXAMPLES GGML_OPENMP GGML_BLAS GGML_METAL GGML_CUDA GGML_VULKAN
    GGML_HIP GGML_SYCL GGML_RPC GGML_OPENCL GGML_CANN GGML_MUSA GGML_WEBGPU
    GGML_OPENVINO GGML_ZDNN GGML_VIRTGPU GGML_VIRTGPU_BACKEND
    GGML_CPU_KLEIDIAI GGML_CPU_HBM GGML_LLAMAFILE
    GGML_SSE42 GGML_AVX GGML_AVX2 GGML_AVX_VNNI GGML_BMI2 GGML_FMA GGML_F16C
    GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory("${LOCALVQE_SOURCE}/vendor/ggml" ggml EXCLUDE_FROM_ALL)
add_library(speakerdesk_localvqe STATIC
  "${LOCALVQE_SOURCE}/common.cpp" "${LOCALVQE_SOURCE}/model_hash.cpp"
  "${LOCALVQE_SOURCE}/localvqe_graph.cpp" "${LOCALVQE_SOURCE}/localvqe_api.cpp"
  "${LOCALVQE_SOURCE}/daf_frontend.cpp")
target_include_directories(speakerdesk_localvqe PUBLIC "${LOCALVQE_SOURCE}")
target_compile_definitions(speakerdesk_localvqe PUBLIC LOCALVQE_STATIC PRIVATE SPEAKERDESK_STATIC_CPU)
target_link_libraries(speakerdesk_localvqe PUBLIC ggml)
if(MSVC)
  target_compile_definitions(speakerdesk_localvqe PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
  target_compile_options(speakerdesk_localvqe PRIVATE /O2)
else()
  target_compile_options(speakerdesk_localvqe PRIVATE -O2)
  set_source_files_properties("${LOCALVQE_SOURCE}/daf_frontend.cpp" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(speakerdesk_echo PRIVATE speakerdesk_localvqe)

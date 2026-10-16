// Disposable feasibility runner: graph math and pinned weights live in the ONNX artifact.
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
using Frame = std::array<float, 1282>;
using Output = std::array<float, 257>;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool latch(bool old, float probability) {
    return probability >= .5f ? true : probability < .35f ? false : old;
}

std::vector<Output> run(Ort::Env& env, const std::filesystem::path& directory,
                        const std::vector<Frame>& frames, bool dml, int adapter) {
    const char* label = dml ? "dml" : "cpu";
    Ort::SessionOptions options;
    options.DisableMemPattern();
    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.EnableProfiling((directory / (dml ? L"dml-profile" : L"cpu-profile")).c_str());
    if (dml) {
        const OrtDmlApi* api = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
            "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&api)));
        Ort::ThrowOnError(api->SessionOptionsAppendExecutionProvider_DML(options, adapter));
    }
    auto started = Clock::now();
    Ort::Session session(env, (directory / L"silero.onnx").c_str(), options);
    double load_ms = elapsed(started);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 3> audio_shape = {1, 1, 640};
    const std::array<int64_t, 2> state_shape = {1, 128};
    const char* inputs[] = {"audio", "hidden", "cell"};
    const char* outputs[] = {"probability", "hidden_out", "cell_out"};
    std::vector<Output> previous;
    for (int pass = 0; pass < 2; ++pass) {
        std::array<std::array<float, 128>, 2> hidden{}, cell{};
        std::vector<Output> actual;
        bool actual_latch = false, expected_latch = false;
        float reference_error = 0, reset_error = 0;
        unsigned changed_decisions = 0;
        started = Clock::now();
        for (size_t index = 0; index < frames.size(); ++index) {
            std::array<float, 2> probabilities{};
            for (size_t view = 0; view < 2; ++view) {
                std::array<Ort::Value, 3> tensors = {
                    Ort::Value::CreateTensor<float>(memory,
                        const_cast<float*>(frames[index].data() + view * 640), 640,
                        audio_shape.data(), audio_shape.size()),
                    Ort::Value::CreateTensor<float>(memory, hidden[view].data(), 128,
                        state_shape.data(), state_shape.size()),
                    Ort::Value::CreateTensor<float>(memory, cell[view].data(), 128,
                        state_shape.data(), state_shape.size())};
                auto values = session.Run(Ort::RunOptions{nullptr}, inputs, tensors.data(), 3,
                                          outputs, 3);
                Output result{};
                result[0] = values[0].GetTensorData<float>()[0];
                std::copy_n(values[1].GetTensorData<float>(), 128, result.data() + 1);
                std::copy_n(values[2].GetTensorData<float>(), 128, result.data() + 129);
                for (float value : result) require(std::isfinite(value), "nonfinite model output");
                require(result[0] >= 0 && result[0] <= 1, "invalid probability");
                std::copy_n(result.data() + 1, 128, hidden[view].data());
                std::copy_n(result.data() + 129, 128, cell[view].data());
                probabilities[view] = result[0];
                reference_error = std::max(reference_error,
                    std::abs(result[0] - frames[index][1280 + view]));
                if (pass) for (size_t element = 0; element < result.size(); ++element)
                    reset_error = std::max(reset_error,
                        std::abs(result[element] - previous[actual.size()][element]));
                actual.push_back(result);
            }
            actual_latch = latch(actual_latch, std::max(probabilities[0], probabilities[1]));
            expected_latch = latch(expected_latch,
                std::max(frames[index][1280], frames[index][1281]));
            changed_decisions += actual_latch != expected_latch;
        }
        std::cout << "{\"provider_requested\":\"" << label << "\",\"pass\":" << pass
            << ",\"load_ms\":" << load_ms << ",\"run_ms\":" << elapsed(started)
            << ",\"frames_per_view\":" << frames.size()
            << ",\"max_probability_error_vs_mlx\":" << reference_error
            << ",\"speech_decision_changes\":" << changed_decisions
            << ",\"max_reset_error\":" << reset_error << "}" << std::endl;
        // The repository documents MLX backend drift; speech decisions must still agree.
        require(reference_error <= .01f, "probability drift exceeds 0.01 vs independent MLX reference");
        require(changed_decisions == 0, "speech decisions differ from independent reference");
        require(reset_error <= 1e-5f, "recurrent state reset is not repeatable");
        previous = std::move(actual);
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto profile = session.EndProfilingAllocated(allocator);
    std::cout << label << " profile: " << profile.get() << std::endl;
    return previous;
}

int wmain(int argc, wchar_t** argv) {
    try {
        require(argc == 3, "usage: silero_ort_probe.exe <artifact-directory> <DXGI-adapter-index>");
        std::filesystem::path directory(argv[1]);
        int adapter = std::stoi(argv[2]);
        require(adapter >= 0, "adapter index must be nonnegative");
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        require(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))), "DXGI factory creation failed");
        Microsoft::WRL::ComPtr<IDXGIAdapter> hardware;
        require(SUCCEEDED(factory->EnumAdapters(adapter, &hardware)), "DXGI adapter is unavailable");
        DXGI_ADAPTER_DESC description{};
        require(SUCCEEDED(hardware->GetDesc(&description)), "DXGI adapter description failed");
        char adapter_name[512]{};
        require(WideCharToMultiByte(CP_UTF8, 0, description.Description, -1,
            adapter_name, sizeof(adapter_name), nullptr, nullptr) > 0, "adapter name conversion failed");
        std::cout << "DXGI adapter " << adapter << ": " << adapter_name
            << ", vendor " << description.VendorId << ", device " << description.DeviceId << std::endl;
        std::ifstream input(directory / L"frames.bin", std::ios::binary);
        char magic[8];
        uint32_t count = 0;
        input.read(magic, 8);
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        require(input && std::memcmp(magic, "SDVAD001", 8) == 0 && count == 285,
                "unexpected fixture header");
        std::vector<Frame> frames(count);
        input.read(reinterpret_cast<char*>(frames.data()), frames.size() * sizeof(Frame));
        require(input && input.peek() == EOF, "truncated or trailing fixture data");
        std::cout << "ORT version: " << OrtGetApiBase()->GetVersionString() << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "speakerdesk-silero-feasibility");
        auto cpu = run(env, directory, frames, false, adapter);
        auto dml = run(env, directory, frames, true, adapter);
        float probability_error = 0, state_error = 0;
        for (size_t step = 0; step < cpu.size(); ++step) {
            probability_error = std::max(probability_error, std::abs(cpu[step][0] - dml[step][0]));
            for (size_t element = 1; element < cpu[step].size(); ++element)
                state_error = std::max(state_error,
                    std::abs(cpu[step][element] - dml[step][element]));
        }
        std::cout << "{\"comparison\":\"cpu_vs_dml\",\"max_probability_error\":"
            << probability_error << ",\"max_recurrent_state_error\":" << state_error << "}" << std::endl;
        require(probability_error <= .001f && state_error <= .01f, "CPU/DML numerical disagreement");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

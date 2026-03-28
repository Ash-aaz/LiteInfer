#include "inference_engine.h"
#include <iostream>

    InferenceEngine::InferenceEngine(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "LiteInferEngine"), 
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}),
          output_buffer(10, 0.0f)
    {
        std::cout << "Model loaded successfully!" << std::endl;
    }

    const std::vector<float>& InferenceEngine::get_predictions() {
        return output_buffer;
    }

    void InferenceEngine::forward_pass(const std::vector<float>& input_frame) {
        const float* input = input_frame.data();
        float* output = output_buffer.data();

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
        );

        std::vector<int64_t> input_shape = {1, 1, 28, 28};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input),
            input_frame.size(),
            input_shape.data(),
            input_shape.size()
        );

        std::vector<int64_t> output_shape = {1, 10};
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            output,
            output_buffer.size(),
            output_shape.data(),
            output_shape.size()
        );

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        const char* raw_input_name = input_name.get();
        const char* raw_output_name = output_name.get();

        session.Run(
            Ort::RunOptions(nullptr),
            &raw_input_name,
            &input_tensor,
            1,
            &raw_output_name,
            &output_tensor,
            1
        );
    }
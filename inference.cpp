#include "inference_engine.h"
#include <iostream>

    InferenceEngine::InferenceEngine(const std::string& model_path, Device dev) 
        : env(ORT_LOGGING_LEVEL_WARNING, "LiteInferEngine"),
          device(dev),
          session_options(create_session_options(dev)), 
          session(env, model_path.c_str(), session_options),
          output_buffer(10, 0.0f)
    {
        std::string device_name = (dev == Device::CUDA) ? "CUDA" : "CPU";
        std::cout << device_name << " Engine loaded successfully!" << std::endl;
    }

    const std::vector<float>& InferenceEngine::get_predictions() {
        return output_buffer;
    }

    Ort::SessionOptions InferenceEngine::create_session_options(Device dev) {

        if (dev == Device::CUDA) {
            Ort::SessionOptions opts;
            OrtCUDAProviderOptions cuda_options;

            cuda_options.device_id = 0;
            cuda_options.gpu_mem_limit = 1024 * 1024 * 1024;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;

            opts.AppendExecutionProvider_CUDA(cuda_options);
            return opts;
        }
        else if (dev == Device::CPU) {
            Ort::SessionOptions default_opts;
            return default_opts;
        }
        else {
            std::cout << "Invalid inference mode entered. Defaulted to CPU" << "\n";
            Ort::SessionOptions default_opts;
            return default_opts;
        }
    }

    void InferenceEngine::forward_pass(const std::vector<float>& input_frame) {
        const float* input = input_frame.data();
        float* output = output_buffer.data();

        auto memory_info = (device == Device::CUDA) ? 
            Ort::MemoryInfo("Cuda",
                OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault) : 
            Ort::MemoryInfo::CreateCpu(
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
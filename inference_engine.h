#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

enum class Device {CPU, CUDA};
class InferenceEngine {
private:
    Ort::Env env;
    Device device;
    Ort::SessionOptions session_options;
    static Ort::SessionOptions create_session_options(Device dev);
    Ort::Session session;
    std::vector<float> output_buffer;

public:
    InferenceEngine(const std::string& model_path, Device dev);
    const std::vector<float>& get_predictions();
    void forward_pass(const std::vector<float>& input_frame);
};

#endif
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "inference_engine.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>

std::vector<float> load_image(const std::string& model_path) {
    int width, height, channels;
    auto raw_info = stbi_load(model_path.c_str(), &width, &height, &channels, 1);
    std::vector<float> image_pixels;
    image_pixels.reserve(height * width);

    if (width == 28 && height == 28) {
        for (int i = 0; i < 28*28; i++) {
            image_pixels.push_back(raw_info[i] / 255.0);
        }
        stbi_image_free(raw_info);
        return image_pixels;
    }
    else {
        std::cout << "Image size not loaded properly. Expected: 28 * 28 Recieved:" << width << " * " << height <<"\n";
        stbi_image_free(raw_info);

        std::vector<float> incorrect_image;
        return incorrect_image;
    }
}

int main() {
    InferenceEngine gpu_engine("model/mnist-8.onnx", Device::CUDA);
    InferenceEngine cpu_engine("model/mnist-8.onnx", Device::CPU);

    std::vector<float> image_pixels = load_image("test_image/60000.png");

    // warmup both to increase latency accuracy
    gpu_engine.forward_pass(image_pixels);
    cpu_engine.forward_pass(image_pixels);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    gpu_engine.forward_pass(image_pixels);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_engine.forward_pass(image_pixels);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);

    std::vector<float> outputs = gpu_engine.get_predictions();
    float max_output = *std::max_element(outputs.begin(), outputs.end());

    std::vector<float> predictions;
    predictions.reserve(outputs.size());

    float total_exponentiated_value = 0;

    for (int i = 0; i < outputs.size(); i++) {
        total_exponentiated_value += exp(outputs[i] - max_output);
    }

    for (int j = 0; j < outputs.size(); j++) {
        predictions.push_back((exp(outputs[j] - max_output))/total_exponentiated_value);
    }

    for (int k = 0; k < 10; ++k) {
        std::cout << k << ": " << std::fixed << std::setprecision(2) << predictions[k] * 100 << "\n";
    }
    std::cout << "GPU Latency: " << duration_gpu.count() << "µs" << "\n";
    std::cout << "CPU Latency: " << duration_cpu.count() << "µs" << "\n";
    return 0;
}


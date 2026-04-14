#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "inference_engine.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

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
    InferenceEngine engine("model/mnist-8.onnx");
    std::vector<float> image_pixels = load_image("test_image/60000.png");

    engine.forward_pass(image_pixels);
    std::vector<float> outputs = engine.get_predictions();
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
    return 0;
}


#include "inference_engine.h"
#include <iostream>

int main() {
    InferenceEngine engine("model/mnist-8.onnx");
    std::vector<float> dummy_input(784, 0.0f);

    engine.forward_pass(dummy_input);

    auto outputs = engine.get_predictions();

    for(int i =0; i < 10; ++i) {
        std::cout << i << ": " << outputs[i] << "\n";
    }

    return 0;
}


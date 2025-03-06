#include <iostream>
#include </usr/local/include/onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <cassert>

// Convert float32 to float16 (IEEE 754 Half-precision)
uint16_t float32_to_float16(float value) {
    uint32_t f = *(uint32_t*)&value;
    uint16_t h = ((f >> 16) & 0x8000) | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f >> 13) & 0x03ff);
    return h;
}

int main() {
    // 1. Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // 2. Create session options
    Ort::SessionOptions session_options;

    // 3. Load .ort model
    std::string model_path = "optimized_model.ort";
    Ort::Session session(env, model_path.c_str(), session_options);

    std::cout << "✅ ORT model loaded successfully!" << std::endl;

    // 4. Get input information
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "📌 Input name: " << input_name.get() << std::endl;
    std::cout << "📌 Input shape: [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        std::cout << input_shape[i] << (i < input_shape.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    // 5. Convert float32 to float16 and create input tensor
    std::vector<Ort::Float16_t> input_tensor(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]);
    for (size_t i = 0; i < input_tensor.size(); i++) {
        uint16_t f16 = float32_to_float16(1.0f);
        input_tensor[i].val = f16;
    }

    // 6. Create ONNX Runtime tensor with float16 data
    std::vector<int64_t> input_dims = {1, 512, 180, 180};  // Model expects float16 tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor_value = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());

    assert(input_tensor_value.IsTensor());

    // 7. Run inference
    // 获取输出数量
    size_t num_outputs = session.GetOutputCount();
    std::cout << "📌 Number of outputs: " << num_outputs << std::endl;

    // 准备输出名称
    std::vector<const char*> output_names(num_outputs);
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    for (size_t i = 0; i < num_outputs; i++) {
        output_names_ptr.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names[i] = output_names_ptr.back().get();
        std::cout << "📌 Output " << i << " name: " << output_names[i] << std::endl;
    }

    const char* input_names[] = {input_name.get()};
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor_value, 
        1, 
        output_names.data(), 
        num_outputs
    );

    // 8. 处理所有输出
    for (size_t i = 0; i < num_outputs; i++) {
        auto output_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "📌 Output " << i << " shape: [";
        for (size_t j = 0; j < output_shape.size(); j++) {
            std::cout << output_shape[j] << (j < output_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // 获取输出数据
        auto* output_data = output_tensors[i].GetTensorMutableData<float>();

        // 打印前5个值
        std::cout << "🟢 Output " << i << " first 5 values: ";
        for (int j = 0; j < 5; j++) {
            std::cout << output_data[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

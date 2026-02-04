#include "../../../include/llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../ops.hpp"
#include "../llaisys_tensor.hpp" 
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>

// --- 跨平台兼容性宏 ---
#ifdef _WIN32
#define LLAISYS_EXPORT __declspec(dllexport)
#else
#define LLAISYS_EXPORT 
#endif

using namespace llaisys;

// 定义模型结构
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    std::vector<LlaisysTensor*> wrapper_keeper;

    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    // Buffers
    LlaisysTensor* x_buf_wrapper;
    LlaisysTensor* residual_wrapper;
    LlaisysTensor* logits_buf_wrapper;
    
    LlaisysTensor* pos_id_wrapper;
    LlaisysTensor* input_id_wrapper;

    int64_t current_pos;

    LlaisysQwen2Model() : current_pos(0) {}

    ~LlaisysQwen2Model() {
        for (auto p : wrapper_keeper) delete p;
        if (weights.attn_norm_w) delete[] weights.attn_norm_w;
        if (weights.attn_q_w) delete[] weights.attn_q_w;
        if (weights.attn_q_b) delete[] weights.attn_q_b;
        if (weights.attn_k_w) delete[] weights.attn_k_w;
        if (weights.attn_k_b) delete[] weights.attn_k_b;
        if (weights.attn_v_w) delete[] weights.attn_v_w;
        if (weights.attn_v_b) delete[] weights.attn_v_b;
        if (weights.attn_o_w) delete[] weights.attn_o_w;
        if (weights.mlp_norm_w) delete[] weights.mlp_norm_w;
        if (weights.mlp_gate_w) delete[] weights.mlp_gate_w;
        if (weights.mlp_up_w) delete[] weights.mlp_up_w;
        if (weights.mlp_down_w) delete[] weights.mlp_down_w;
    }
};

// --- 内部辅助函数 (C++ Linkage) ---

// 必须在 extern "C" 外，因为它返回 std::shared_ptr (tensor_t)
inline tensor_t unwrap(llaisysTensor_t ptr) {
    if (!ptr) return nullptr;
    return ((LlaisysTensor*)ptr)->tensor;
}

llaisysTensor_t create_wrapped(LlaisysQwen2Model* model, const std::vector<size_t>& shape, llaisysDataType_t dtype, llaisysDeviceType_t dev, int dev_id) {
    auto t = Tensor::create(shape, dtype, dev, dev_id);
    LlaisysTensor* wrapper = new LlaisysTensor();
    wrapper->tensor = t;
    model->wrapper_keeper.push_back(wrapper);
    return (llaisysTensor_t)wrapper;
}

// --- 导出接口 (C Linkage) ---
extern "C" {

LLAISYS_EXPORT LlaisysQwen2Model* llaisysQwen2ModelCreate(const LlaisysQwen2Meta* meta, llaisysDeviceType_t dev, int* device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    int dev_id = (ndevice > 0) ? device_ids[0] : 0;
    auto cw = [&](std::vector<size_t> shape) -> llaisysTensor_t {
        return create_wrapped(model, shape, (llaisysDataType_t)meta->dtype, dev, dev_id);
    };

    size_t nl = meta->nlayer;
    model->weights.attn_norm_w = new llaisysTensor_t[nl];
    model->weights.attn_q_w = new llaisysTensor_t[nl];
    model->weights.attn_q_b = new llaisysTensor_t[nl];
    model->weights.attn_k_w = new llaisysTensor_t[nl];
    model->weights.attn_k_b = new llaisysTensor_t[nl];
    model->weights.attn_v_w = new llaisysTensor_t[nl];
    model->weights.attn_v_b = new llaisysTensor_t[nl];
    model->weights.attn_o_w = new llaisysTensor_t[nl];
    model->weights.mlp_norm_w = new llaisysTensor_t[nl];
    model->weights.mlp_gate_w = new llaisysTensor_t[nl];
    model->weights.mlp_up_w = new llaisysTensor_t[nl];
    model->weights.mlp_down_w = new llaisysTensor_t[nl];

    model->weights.in_embed = cw({meta->voc, meta->hs});
    model->weights.out_embed = cw({meta->voc, meta->hs});
    model->weights.out_norm_w = cw({meta->hs});

    for(size_t i=0; i<nl; i++) {
        model->weights.attn_norm_w[i] = cw({meta->hs});
        model->weights.attn_q_w[i] = cw({meta->nh * meta->dh, meta->hs});
        model->weights.attn_q_b[i] = cw({meta->nh * meta->dh});
        model->weights.attn_k_w[i] = cw({meta->nkvh * meta->dh, meta->hs});
        model->weights.attn_k_b[i] = cw({meta->nkvh * meta->dh});
        model->weights.attn_v_w[i] = cw({meta->nkvh * meta->dh, meta->hs});
        model->weights.attn_v_b[i] = cw({meta->nkvh * meta->dh});
        model->weights.attn_o_w[i] = cw({meta->hs, meta->nh * meta->dh});
        model->weights.mlp_norm_w[i] = cw({meta->hs});
        model->weights.mlp_gate_w[i] = cw({meta->di, meta->hs});
        model->weights.mlp_up_w[i] = cw({meta->di, meta->hs});
        model->weights.mlp_down_w[i] = cw({meta->hs, meta->di});
        
        model->k_cache.push_back(Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, (llaisysDataType_t)meta->dtype, dev, dev_id));
        model->v_cache.push_back(Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, (llaisysDataType_t)meta->dtype, dev, dev_id));
    }
    
    model->x_buf_wrapper = (LlaisysTensor*)create_wrapped(model, {1, meta->hs}, (llaisysDataType_t)meta->dtype, dev, dev_id);
    model->residual_wrapper = (LlaisysTensor*)create_wrapped(model, {1, meta->hs}, (llaisysDataType_t)meta->dtype, dev, dev_id);
    model->logits_buf_wrapper = (LlaisysTensor*)create_wrapped(model, {1, meta->voc}, (llaisysDataType_t)meta->dtype, dev, dev_id);
    model->pos_id_wrapper = (LlaisysTensor*)create_wrapped(model, {1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    model->input_id_wrapper = (LlaisysTensor*)create_wrapped(model, {1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);

    return model;
}

LLAISYS_EXPORT void llaisysQwen2ModelDestroy(LlaisysQwen2Model* model) {
    if (model) delete model;
}

LLAISYS_EXPORT LlaisysQwen2Weights* llaisysQwen2ModelWeights(LlaisysQwen2Model* model) {
    return &model->weights;
}

LLAISYS_EXPORT int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    auto& m = model->meta;
    auto dev = model->x_buf_wrapper->tensor->deviceType();
    
    // 取得当前位置的 Token
    int64_t current_token_id = token_ids[model->current_pos];
    *(int64_t*)model->input_id_wrapper->tensor->data() = current_token_id;
    
    // 1. Embedding 层
    {
        auto in_embed = unwrap(model->weights.in_embed);
        size_t hidden = m.hs;
        size_t elem_size = in_embed->elementSize();
        size_t row_bytes = hidden * elem_size;

        const std::byte* src = in_embed->data() + (size_t)current_token_id * row_bytes;
        std::byte* dst = model->x_buf_wrapper->tensor->data();
        std::memcpy(dst, src, row_bytes);
    }

    // 2. Transformer 层循环
    for(size_t i=0; i<m.nlayer; i++) {
        // 残差连接
        ops::rearrange(model->residual_wrapper->tensor, model->x_buf_wrapper->tensor);

        // 注意力层前 Norm
        ops::rms_norm(model->x_buf_wrapper->tensor, model->x_buf_wrapper->tensor, unwrap(model->weights.attn_norm_w[i]), m.epsilon);

        // 计算 Q, K, V
        auto q = Tensor::create({1, m.nh * m.dh}, (llaisysDataType_t)m.dtype, dev, 0);
        auto k = Tensor::create({1, m.nkvh * m.dh}, (llaisysDataType_t)m.dtype, dev, 0);
        auto v = Tensor::create({1, m.nkvh * m.dh}, (llaisysDataType_t)m.dtype, dev, 0);

        ops::linear(q, model->x_buf_wrapper->tensor, unwrap(model->weights.attn_q_w[i]), unwrap(model->weights.attn_q_b[i]));
        ops::linear(k, model->x_buf_wrapper->tensor, unwrap(model->weights.attn_k_w[i]), unwrap(model->weights.attn_k_b[i]));
        ops::linear(v, model->x_buf_wrapper->tensor, unwrap(model->weights.attn_v_w[i]), unwrap(model->weights.attn_v_b[i]));

        auto q_view = q->view({1, m.nh, m.dh});
        auto k_view = k->view({1, m.nkvh, m.dh});
        auto v_view = v->view({1, m.nkvh, m.dh});

        // RoPE 旋转位置编码
        *(int64_t*)model->pos_id_wrapper->tensor->data() = model->current_pos;
        ops::rope(q_view, q_view, model->pos_id_wrapper->tensor, m.theta);
        ops::rope(k_view, k_view, model->pos_id_wrapper->tensor, m.theta);

        // 更新 KV Cache
        auto k_slot = model->k_cache[i]->slice(0, model->current_pos, model->current_pos + 1);
        auto v_slot = model->v_cache[i]->slice(0, model->current_pos, model->current_pos + 1);
        ops::rearrange(k_slot, k_view);
        ops::rearrange(v_slot, v_view);

        auto k_history = model->k_cache[i]->slice(0, 0, model->current_pos + 1);
        auto v_history = model->v_cache[i]->slice(0, 0, model->current_pos + 1);

        // Self-Attention
        auto attn_out = Tensor::create({1, m.nh * m.dh}, (llaisysDataType_t)m.dtype, dev, 0);
        float scale = 1.0f / sqrtf((float)m.dh); // 显式使用 sqrtf 解决 Windows 警告
        ops::self_attention(attn_out->view({1, m.nh, m.dh}), q_view, k_history, v_history, scale);

        // 注意力输出映射
        ops::linear(model->x_buf_wrapper->tensor, attn_out, unwrap(model->weights.attn_o_w[i]), nullptr);

        // 残差累加
        ops::add(model->x_buf_wrapper->tensor, model->x_buf_wrapper->tensor, model->residual_wrapper->tensor);

        // MLP 层前 Norm
        ops::rearrange(model->residual_wrapper->tensor, model->x_buf_wrapper->tensor);
        ops::rms_norm(model->x_buf_wrapper->tensor, model->x_buf_wrapper->tensor, unwrap(model->weights.mlp_norm_w[i]), m.epsilon);

        // MLP (SwiGLU)
        auto gate = Tensor::create({1, m.di}, (llaisysDataType_t)m.dtype, dev, 0);
        auto up = Tensor::create({1, m.di}, (llaisysDataType_t)m.dtype, dev, 0);
        
        ops::linear(gate, model->x_buf_wrapper->tensor, unwrap(model->weights.mlp_gate_w[i]), nullptr);
        ops::linear(up, model->x_buf_wrapper->tensor, unwrap(model->weights.mlp_up_w[i]), nullptr);
        
        ops::swiglu(gate, gate, up);

        ops::linear(model->x_buf_wrapper->tensor, gate, unwrap(model->weights.mlp_down_w[i]), nullptr);

        // 最终残差累加
        ops::add(model->x_buf_wrapper->tensor, model->x_buf_wrapper->tensor, model->residual_wrapper->tensor);
    }

    // 3. 输出层
    ops::rms_norm(model->x_buf_wrapper->tensor, model->x_buf_wrapper->tensor, unwrap(model->weights.out_norm_w), m.epsilon);
    ops::linear(model->logits_buf_wrapper->tensor, model->x_buf_wrapper->tensor, unwrap(model->weights.out_embed), nullptr);

    // 4. Argmax
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    auto max_val = Tensor::create({1}, (llaisysDataType_t)m.dtype, LLAISYS_DEVICE_CPU, 0);
    ops::argmax(max_idx, max_val, model->logits_buf_wrapper->tensor);

    model->current_pos++;
    return *reinterpret_cast<int64_t*>(max_idx->data());
}

} // extern "C"
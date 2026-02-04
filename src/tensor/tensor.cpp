#include "tensor.hpp"
#include "../utils.hpp"
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = (ptrdiff_t)stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() { return _storage->memory() + _offset; }
const std::byte *Tensor::data() const { return _storage->memory() + _offset; }
size_t Tensor::ndim() const { return _meta.shape.size(); }
const std::vector<size_t> &Tensor::shape() const { return _meta.shape; }
const std::vector<ptrdiff_t> &Tensor::strides() const { return _meta.strides; }
llaisysDataType_t Tensor::dtype() const { return _meta.dtype; }
llaisysDeviceType_t Tensor::deviceType() const { return _storage->deviceType(); }
int Tensor::deviceId() const { return _storage->deviceId(); }

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}
size_t Tensor::elementSize() const { return utils::dsize(_meta.dtype); }

std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor: shape[ ";
    for (auto s : this->shape()) ss << s << " ";
    ss << "] strides[ ";
    for (auto s : this->strides()) ss << s << " ";
    ss << "] dtype=" << this->dtype();
    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    // 修复 C4267：显式转换类型以匹配 Windows 下的 size_t 与 int 比较
    if (dim == (size_t)(shape.size() - 1)) {
        for (size_t i = 0; i < shape[dim]; i++) {
             if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[(size_t)(i * strides[dim])]) << " ";
            } else {
                std::cout << data[(size_t)(i * strides[dim])] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < (size_t)(shape.size() - 1)) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + (size_t)(i * strides[dim]), shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64: return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    default: std::cout << "[Data Print Not Supported for this Dtype]" << std::endl;
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    }
}

// ==========================================
// 作业 #1 核心实现 (修复版)
// ==========================================

bool Tensor::isContiguous() const {
    size_t z = 1;
    // 修复 C4267：Windows MSVC 严禁 size_t 与 int 混用
    for (int i = (int)(_meta.shape.size() - 1); i >= 0; i--) {
        if (_meta.strides[(size_t)i] != (ptrdiff_t)z) return false;
        z *= _meta.shape[(size_t)i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != ndim()) throw std::runtime_error("permute dim mismatch");
    TensorMeta new_meta = _meta;
    for(size_t i=0; i<ndim(); ++i) {
        new_meta.shape[i] = _meta.shape[order[i]];
        new_meta.strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for(auto s : shape) new_numel *= s;
    if(new_numel != numel()) throw std::runtime_error("view numel mismatch");
    
    if (!isContiguous()) throw std::runtime_error("view on non-contiguous tensor not supported in HW1");

    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for(int i = (int)(shape.size() - 1); i >= 0; --i) {
        new_strides[(size_t)i] = (ptrdiff_t)stride;
        stride *= shape[(size_t)i];
    }
    
    TensorMeta new_meta = _meta;
    new_meta.shape = shape;
    new_meta.strides = new_strides;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= ndim()) throw std::out_of_range("slice dim out of range");
    if (end > shape()[dim] || start >= end) throw std::out_of_range("slice index invalid");
    
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;
    size_t new_offset = _offset + start * (size_t)_meta.strides[dim] * elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    if (!isContiguous()) throw std::runtime_error("load non-contiguous not supported");
    
    size_t size = numel() * elementSize();
    core::context().setDevice(deviceType(), deviceId());
    
    auto kind = (deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(data(), src_, size, kind);
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    throw std::runtime_error("contiguous() unimplemented for non-contiguous tensor");
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    return view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device_type == this->deviceType()) return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    throw std::runtime_error("tensor.to() cross-device copy unimplemented in HW1 scope");
}

} // namespace llaisys
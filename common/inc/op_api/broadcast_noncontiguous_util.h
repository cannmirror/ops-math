/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_noncontiguous_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_BROADCAST_NONCONTIGUOUS_UTIL_H_
#define CANN_OPS_BUILT_IN_BROADCAST_NONCONTIGUOUS_UTIL_H_

#include "op_api/aclnn_check.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

namespace op {

static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t DATA_SIZE_LIMIT = 8192 * 64;
static constexpr int64_t DIM_TWO = 2;
static constexpr int64_t DIM_THREE = 3;
static constexpr int64_t DIM_FOUR = 4;
static constexpr int64_t LAST_TRANSPOSE_LONG_AXIS_LIMIT = 512;

static bool IsOnlyLastTwoAxesTransposed(const op::Shape& viewShape, const op::Strides& strides)
{
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = strides.size();
    if (viewShape.GetDimNum() < DIM_TWO || shapeDim != stridesDim) {
        return false;
    }
    size_t lastDim = shapeDim - 1;
    size_t secondLastDim = stridesDim - DIM_TWO;
    bool transposedStride = (strides[lastDim] == viewShape[secondLastDim]) && (strides[secondLastDim] == 1);
    bool othersContiguous = true;
    if (shapeDim > 2) {
        int64_t expectedStride = viewShape[lastDim] * viewShape[secondLastDim];
        for (int64_t i = shapeDim - DIM_THREE; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                othersContiguous = false;
                break;
            }
            expectedStride = expectedStride * viewShape[i];
        }
    }
    bool result = transposedStride && othersContiguous;
    return result;
}

static bool CheckBasicConstraints(const op::Shape& viewShape, const op::Strides& viewStride)
{
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = viewStride.size();
    if (shapeDim != stridesDim) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. shapeDim: %d stridesDim: %d", shapeDim, stridesDim);
        return false;
    }
    if (shapeDim > DIM_FOUR) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. shapeDim: %d > 4", shapeDim);
        return false;
    }
    if (!IsRegBase()) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. not RegBase");
        return false;
    }
    return true;
}

static bool IsSupportedByLargeLastDim(const op::Shape& viewShape, const op::Strides& viewStride, int64_t cacheLineDim)
{
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = viewStride.size();
    int64_t largeLastDimThreshold = cacheLineDim * 8;
    
    if (stridesDim < 2) {
        return false;
    }
    
    int64_t lastDimSize = viewShape[shapeDim - 1];
    int64_t lastStride = viewStride[stridesDim - 1];
    int64_t secondLastStride = viewStride[stridesDim - DIM_TWO];
    
    if (lastDimSize >= largeLastDimThreshold && lastStride == 1) {
        if (secondLastStride > largeLastDimThreshold * 100) {
            return false;
        }
        if (secondLastStride >= lastDimSize && secondLastStride > largeLastDimThreshold * 10) {
            return false;
        }
        OP_LOGI("Broadcast Template NonContiguous Supported. Shape[-1]: %d > %d and Stride[-1]: 1 Case",
                lastDimSize, largeLastDimThreshold);
        return true;
    }
    return false;
}

static bool IsSupportedByStridePattern(const op::Shape& viewShape, const op::Strides& viewStride, int64_t cacheLineDim)
{
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = viewStride.size();
    int64_t stridePatternThreshold = cacheLineDim * 4;
    if (stridesDim > 1 && viewStride[stridesDim - 1] == 1 && viewShape[shapeDim - 1] < cacheLineDim &&
        viewStride[stridesDim - DIM_TWO] > stridePatternThreshold) {
        OP_LOGI("Broadcast Template NonContiguous Supported. Stride[-1]: 1 and Stride[-2]: %d and Shape[-1]: %d < CacheLineDim: %d Case",
                viewStride[stridesDim - DIM_TWO], viewShape[shapeDim - 1], cacheLineDim);
        return true;
    }
    return false;
}

static bool IsSupportedBySmallDataSize(
    const op::Shape& viewShape, const op::Strides& viewStride, int64_t cacheLineDim, size_t typeSize)
{
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = viewStride.size();
    int64_t dataSize = viewShape.GetShapeSize() * typeSize;
    
    int64_t smallDataSizeThreshold = cacheLineDim;
    int64_t smallDataSizeLimit = DATA_SIZE_LIMIT / 8;
    
    if ((viewShape[shapeDim - 1] < smallDataSizeThreshold) && 
        (dataSize < smallDataSizeLimit) && 
        (viewStride[stridesDim - 1] == 1)) {
        OP_LOGI("Broadcast Template NonContiguous Supported. Stride[-1]: %d Shape[-1]: %d < CacheLineDim: %d and Tensor DataSize: %d < %d Case",
                viewStride[stridesDim - 1], viewShape[shapeDim - 1], cacheLineDim, dataSize, smallDataSizeLimit);
        return true;
    }
    return false;
}

static bool IsLastTransposePreferContiguous(
    const op::Shape& viewShape, const op::Strides& viewStride, int64_t cacheLineDim)
{
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = viewStride.size();
    if (shapeDim < DIM_TWO || shapeDim != stridesDim) {
        return false;
    }
    int64_t secondLastDimSize = viewShape[shapeDim - DIM_TWO];
    int64_t lastDimSize = viewShape[shapeDim - 1];
    
    if (secondLastDimSize < cacheLineDim) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. LastTwoAxesTransposed prefer contiguous, Shape[-2]: %ld < CacheLineDim: %ld",
                secondLastDimSize, cacheLineDim);
        return true;
    }
    if (lastDimSize >= LAST_TRANSPOSE_LONG_AXIS_LIMIT) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. LastTwoAxesTransposed prefer contiguous, Shape[-1]: %ld >= %ld",
                lastDimSize, LAST_TRANSPOSE_LONG_AXIS_LIMIT);
        return true;
    }
    if (lastDimSize < cacheLineDim) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. LastTwoAxesTransposed prefer contiguous, Shape[-1]: %ld < CacheLineDim: %ld",
                lastDimSize, cacheLineDim);
        return true;
    }
    if (secondLastDimSize >= cacheLineDim && lastDimSize >= cacheLineDim && lastDimSize < LAST_TRANSPOSE_LONG_AXIS_LIMIT) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. LastTwoAxesTransposed prefer contiguous, Shape[-2]: %ld >= CacheLineDim: %ld and Shape[-1]: %ld >= CacheLineDim: %ld and Shape[-1]: %ld < %ld",
                secondLastDimSize, cacheLineDim, lastDimSize, cacheLineDim, lastDimSize, LAST_TRANSPOSE_LONG_AXIS_LIMIT);
        return true;
    }
    return false;
}

static bool IsSupportedByTransposedAxes(
    const op::Shape& viewShape, const op::Strides& viewStride, int64_t cacheLineDim)
{
    bool isOnlyLastTwoAxesTransposed = IsOnlyLastTwoAxesTransposed(viewShape, viewStride);
    
    if (isOnlyLastTwoAxesTransposed) {
        bool preferContiguous = IsLastTransposePreferContiguous(viewShape, viewStride, cacheLineDim);
        
        if (preferContiguous) {
            return false;
        }
        OP_LOGI("BroadcastTemplateNonContiguousSupport is True, LastTwoAxesTransposed Case");
        return true;
    }
    return false;
}

static bool IsBroadcastTemplateNonContiguousSupport(const aclTensor* input)
{
    auto viewShape = input->GetViewShape();
    auto viewStride = input->GetViewStrides();
    size_t typeSize = op::TypeSize(input->GetDataType());
    
    if (typeSize == 0) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. typeSize is 0");
        return false;
    }
    int64_t cacheLineDim = CACHE_LINE_SIZE / typeSize;

    if (!CheckBasicConstraints(viewShape, viewStride)) {
        return false;
    }
    
    bool isContiguous = op::IsContiguous(input);
    if (isContiguous) {
        OP_LOGI("Broadcast Template NonContiguous Supported. Tensor is Contiguous");
        return true;
    }
    
    bool largeLastDim = IsSupportedByLargeLastDim(viewShape, viewStride, cacheLineDim);
    if (largeLastDim) {
        return true;
    }
    
    bool stridePattern = IsSupportedByStridePattern(viewShape, viewStride, cacheLineDim);
    if (stridePattern) {
        return true;
    }
    
    bool smallDataSize = IsSupportedBySmallDataSize(viewShape, viewStride, cacheLineDim, typeSize);
    if (smallDataSize) {
        return true;
    }
    
    bool transposedAxes = IsSupportedByTransposedAxes(viewShape, viewStride, cacheLineDim);
    if (transposedAxes) {
        return true;
    }
    
    return false;
}

} // namespace op

#endif // CANN_OPS_BUILT_IN_BROADCAST_NONCONTIGUOUS_UTIL_H_

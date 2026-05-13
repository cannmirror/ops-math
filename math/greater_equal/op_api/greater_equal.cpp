/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "greater_equal.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GreaterEqual);

// 仅1971支持DT_BF16
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16,
    op::DataType::DT_INT64};

// 610lite支持类型
static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8};

// 950支持类型
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
  op::DataType::DT_UINT8, op::DataType::DT_BF16, op::DataType::DT_INT64,
  op::DataType::DT_UINT64, op::DataType::DT_BOOL};

static inline bool IsAiCoreSupport(const aclTensor *self) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    return CheckType(self->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
  }
  if (npuArch == NpuArch::DAV_3102) {
    return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
  }
  // 只需要判断dtype
  return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool IsGeTensorSupportNonContiguous(const aclTensor* input)
{
    auto viewShape = input->GetViewShape();
    auto viewStride = input->GetViewStrides();
    size_t typeSize = op::TypeSize(input->GetDataType());
    
    if (typeSize == 0) {
        OP_LOGI("GeTensor NonContiguous UnSupported. typeSize is 0");
        return false;
    }
    
    size_t shapeDim = viewShape.GetDimNum();
    size_t stridesDim = viewStride.size();
    if (shapeDim != stridesDim) {
        OP_LOGI("GeTensor NonContiguous UnSupported. shapeDim: %d != stridesDim: %d", shapeDim, stridesDim);
        return false;
    }
    if (shapeDim > 4) {
        OP_LOGI("GeTensor NonContiguous UnSupported. shapeDim: %d > 4", shapeDim);
        return false;
    }
    if (!IsRegBase()) {
        OP_LOGI("GeTensor NonContiguous UnSupported. not RegBase");
        return false;
    }
    if (op::IsContiguous(input)) {
        OP_LOGI("GeTensor NonContiguous Supported. tensor is contiguous");
        return true;
    }
    if (viewStride[stridesDim - 1] != 1) {
        OP_LOGI("GeTensor NonContiguous UnSupported. stride[-1]: %d != 1", viewStride[stridesDim - 1]);
        return false;
    }

    int64_t cacheLineDim = 128 / typeSize;
    if (viewShape[shapeDim - 1] >= cacheLineDim) {
        OP_LOGI("GeTensor NonContiguous Supported. large last dim, shape[-1]: %d >= cacheLineDim: %d",
                viewShape[shapeDim - 1], cacheLineDim);
        return true;
    }
    OP_LOGI("GeTensor NonContiguous UnSupported. shape[-1]: %d < cacheLineDim: %d",
            viewShape[shapeDim - 1], cacheLineDim);
    return false;
}

bool IsGreaterEqualSupportNonContiguous(const aclTensor* self) {
    bool selfNonContiguousSupport = IsGeTensorSupportNonContiguous(self);
    bool selfAiCoreSupport = IsAiCoreSupport(self);
    OP_LOGI(
        "IsGreaterEqualSupportNonContiguous: selfNonContiguousSupport %d selfAiCoreSupport %d",
        selfNonContiguousSupport, selfAiCoreSupport);
    return selfNonContiguousSupport && selfAiCoreSupport;
}

const aclTensor *GreaterEqual(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  L0_DFX(GreaterEqual, self, other);

  op::Shape outShape;
  if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
    return nullptr;
  }

  auto out = executor->AllocTensor(outShape, op::DataType::DT_BOOL);
  auto ret = ACL_SUCCESS;

  if (IsAiCoreSupport(self)) {
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GreaterEqual, OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterEqual AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  } else {
    static internal::AicpuTaskSpace space("GreaterEqual");
    ret = ADD_TO_LAUNCHER_LIST_AICPU(GreaterEqual, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterEqual AiCPU ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  }
  return out;
}
} // l0op

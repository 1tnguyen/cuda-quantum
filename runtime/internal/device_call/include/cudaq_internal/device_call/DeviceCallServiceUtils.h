/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string_view>

struct cudaq_realtime_device_call_service;

namespace cudaq_internal::device_call {

using DeviceCallServiceFactoryFn =
    int (*)(cudaq_realtime_device_call_service *out);

DeviceCallServiceFactoryFn
resolveDeviceCallServiceFactory(void *symbolScope,
                                std::string_view servicePostfix = {});

} // namespace cudaq_internal::device_call

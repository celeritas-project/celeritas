/*-----------------------------------*-C-*-------------------------------------
 * Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
 * See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *---------------------------------------------------------------------------*/
/*!
 * \file corecel/device_runtime_api.h
 * \brief Include CUDA or HIP runtime APIs for compiling with host/cc compiler.
 * \deprecated This file should be replaced by "corecel/DeviceRuntimeApi.hh".
 */
// DEPRECATED: remove in Celeritas v1.0
//---------------------------------------------------------------------------//
#ifndef CELERITAS_DEVICE_RUNTIME_API_H
#define CELERITAS_DEVICE_RUNTIME_API_H

#if __cplusplus < 201703L
#    error "Celeritas requires C++17 or greater and is not C compatible"
#endif

#if __GNUC__ > 8 || __clang__
#    pragma GCC warning \
        "corecel/device_runtime_api.h is deprecated and should be replaced by \"corecel/DeviceRuntimeApi.hh\""
#endif

#include "DeviceRuntimeApi.hh"

#endif /* CELERITAS_DEVICE_RUNTIME_API_H */

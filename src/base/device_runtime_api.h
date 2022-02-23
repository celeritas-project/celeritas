//-----------------------------------*-C-*-----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file base/device_runtime_api.h
//! \brief Include CUDA or HIP runtime APIs for compiling with host/cc compiler
//---------------------------------------------------------------------------//
#ifndef CELERITAS_DEVICE_RUNTIME_API_H
#define CELERITAS_DEVICE_RUNTIME_API_H

#include "celeritas_config.h"

#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#elif CELERITAS_USE_HIP
#    ifndef __HIPCC__
#        define __HIP_PLATFORM_AMD__ 1
#        define __HIP_PLATFORM_HCC__ 1
#    endif
#    include <hip/hip_runtime.h>
#endif

#endif // CELERITAS_DEVICE_RUNTIME_API_H

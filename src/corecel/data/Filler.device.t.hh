//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Filler.device.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Filler.hh"

#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/sys/Thrust.device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T, MemSpace M>
void Filler<T, M>::fill_device_impl(Span<T>) const
{
    if (stream_)
    {
        thrust::fill_n(thrust_execute_on(stream_),
                       thrust::device_pointer_cast<T>(data.data()),
                       data.size(),
                       value_);
    }
    else
    {
        thrust::fill_n(thrust_execution_policy<ThrustExecMode::Sync>(),
                       thrust::device_pointer_cast<T>(data.data()),
                       data.size(),
                       value_);
    }
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

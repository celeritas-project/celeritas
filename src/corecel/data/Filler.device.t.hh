//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "corecel/sys/Thrust.device.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T, MemSpace M>
void Filler<T, M>::fill_device_impl(Span<T> data) const
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
        thrust::fill_n(thrust_execute(),
                       thrust::device_pointer_cast<T>(data.data()),
                       data.size(),
                       value_);
    }
    CELER_DEVICE_API_CALL(PeekAtLastError());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

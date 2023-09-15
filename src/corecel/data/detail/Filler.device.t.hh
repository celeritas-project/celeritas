//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/Filler.device.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Thrust.device.hh"

#include "Filler.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T> data) const
{
    thrust::fill_n(thrust_execute_on<ThrustExecMode::Sync>(StreamId{0}),
                   thrust::device_pointer_cast<T>(data.data()),
                   data.size(),
                   value);
    CELER_DEVICE_CHECK_ERROR();
}

template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T> data,
                                             StreamId stream_id) const
{
    thrust::fill_n(thrust_execute_on(stream_id),
                   thrust::device_pointer_cast<T>(data.data()),
                   data.size(),
                   value);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

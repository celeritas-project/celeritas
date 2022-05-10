//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/Filler.cu
//---------------------------------------------------------------------------//
#include "Filler.hh"

#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "corecel/device_runtime_api.h"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void Filler<T, MemSpace::device>::operator()(Span<T> data) const
{
    thrust::fill_n(thrust::device,
                   thrust::device_pointer_cast<T>(data.data()),
                   data.size(),
                   value);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
template class Filler<real_type, MemSpace::device>;
template class Filler<size_type, MemSpace::device>;
template class Filler<int, MemSpace::device>;
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

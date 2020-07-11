//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cu
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
void device_memset(void* data, int fill_value, size_type count)
{
    auto* data_char = static_cast<unsigned char*>(data);

    thrust::uninitialized_fill_n(
        thrust::device,
        thrust::device_pointer_cast<unsigned char>(data_char),
        count,
        static_cast<unsigned int>(fill_value));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

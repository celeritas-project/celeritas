//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Memory.cu
//---------------------------------------------------------------------------//
#include "Memory.hh"

#include <limits>
#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fill the pointed-to data with the given fill value. This function has the
 * same signature as std::memset: the \c fill_value must look like a single
 * byte of memory, and \c count is the size of the data in bytes.
 */
void device_memset(void* data, int fill_value, size_type count)
{
    CELER_EXPECT(fill_value >= 0
                 && fill_value <= std::numeric_limits<unsigned char>::max());
    auto* data_char = static_cast<unsigned char*>(data);

    thrust::uninitialized_fill_n(
        thrust::device,
        thrust::device_pointer_cast<unsigned char>(data_char),
        count,
        static_cast<unsigned char>(fill_value));
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace celeritas

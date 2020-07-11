//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.i.hh
//---------------------------------------------------------------------------//

#include "base/Atomics.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to the storage pointers
 */
CELER_FUNCTION
StackAllocator::StackAllocator(const StackAllocatorView& view) : shared_(view)
{
}

//---------------------------------------------------------------------------//
/*!
 * Allocate like malloc.
 */
CELER_FUNCTION auto StackAllocator::operator()(size_type size) -> result_type
{
    size_type start = atomic_add(data_.size, size);
    if (start + size > data_.capacity)
        return nullptr;
    return data_.data + start;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

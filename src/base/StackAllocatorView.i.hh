//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorView.i.hh
//---------------------------------------------------------------------------//

#include "base/Atomics.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to the storage pointers
 */
CELER_FUNCTION
StackAllocatorView::StackAllocatorView(const StackAllocatorPointers& view)
    : shared_(view)
{
}

//---------------------------------------------------------------------------//
/*!
 * Allocate like malloc.
 */
CELER_FUNCTION auto StackAllocatorView::operator()(size_type size)
    -> result_type
{
    size_type start = atomic_add(shared_.size, size);
    if (start + size > shared_.storage.size())
        return nullptr;
    return shared_.storage.data() + start;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

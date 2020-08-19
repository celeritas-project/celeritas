//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocatorView.i.hh
//---------------------------------------------------------------------------//

#include "base/Atomics.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION SecondaryAllocatorView::SecondaryAllocatorView(
    const SecondaryAllocatorPointers& shared)
    : shared_(shared)
{
    REQUIRE(shared);
}

//---------------------------------------------------------------------------//
/*!
 * Allocate space for this many secondaries.
 *
 * Returns NULL if allocation failed due to out-of-memory. Ensures that the
 * shared size reflects the amount of data allocated
 */
CELER_FUNCTION auto SecondaryAllocatorView::operator()(size_type count)
    -> result_type
{
    // Atomic add 'count' to the shared size
    size_type start = atomic_add(shared_.size, count);
    if (CELER_UNLIKELY(start + count > shared_.storage.size()))
    {
        // Out of memory: restore the old value so that another thread can
        // potentially use it. Multiple threads are likely to exceed the
        // capacity simultaneously. Only one has a "start" value less than or
        // equal to the total capacity: the remainder are (arbitrarily) higher
        // than that.
        if (start <= this->capacity())
        {
            // We were the first thread to exceed capacity, even though other
            // threads might have failed (and might still be failing) to
            // allocate. Restore the actual allocated size to the start value.
            // This might allow another thread with a smaller allocation to
            // succeed, but it also guarantees that at the end of the kernel,
            // the size reflects the actual capacity.
            *shared_.size = start;
        }

        // TODO It might be useful to set an "out of memory" flag to make it
        // easier for host code to detect whether a failure occurred, rather
        // than looping through primaries and testing for failure.

        // Return null pointer, indicating failure to allocate.
        return nullptr;
    }

    // Initialize the data at the newly "allocated" address
    return new (shared_.storage.data() + start) Secondary[count];
}

//---------------------------------------------------------------------------//
/*!
 * View all allocated secondaries.
 *
 * This cannot be called while any running kernel could be modifiying the size.
 */
CELER_FUNCTION auto SecondaryAllocatorView::capacity() const -> size_type
{
    return shared_.storage.size();
}

//---------------------------------------------------------------------------//
/*!
 * View all allocated secondaries.
 *
 * This cannot be called while any running kernel could be modifiying the size.
 */
CELER_FUNCTION auto SecondaryAllocatorView::secondaries() -> SpanSecondary
{
    REQUIRE(*shared_.size <= this->capacity());
    return {shared_.storage.data(), *shared_.size};
}

//---------------------------------------------------------------------------//
/*!
 * View all allocated secondaries (immutable).
 *
 * This cannot be called while any running kernel could be modifiying the size.
 */
CELER_FUNCTION auto SecondaryAllocatorView::secondaries() const
    -> constSpanSecondary
{
    REQUIRE(*shared_.size <= this->capacity());
    return {shared_.storage.data(), *shared_.size};
}
//---------------------------------------------------------------------------//
} // namespace celeritas

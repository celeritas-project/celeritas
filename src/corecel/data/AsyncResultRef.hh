//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncResultRef.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "ObserverPtr.hh"
#include "PinnedAllocator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reference to pinned host memory written asynchronously by a stream.
 *
 * This marks a host destination for an asynchronous device-to-host copy. The
 * referenced value \b must be in pinned host memory, which must remain valid
 * until the stream has completed the write. The value must not be read until
 * the stream is synchronized (directly or through an event recorded after the
 * enqueued write).
 *
 * \code
    std::vector<size_type, PinnedAllocator<size_type>> count(1);
    enqueue_count(state, AsyncResultRef{count});
    stream.sync();
    use(count.front());
   \endcode
 */
template<class T>
struct AsyncResultRef
{
    //! Host destination of the asynchronous write
    ObserverPtr<T, MemSpace::host> ptr;

    //! Construct with a null reference
    AsyncResultRef() = default;

    // Construct from the first element of a pinned host vector
    inline explicit AsyncResultRef(std::vector<T, PinnedAllocator<T>>& vec);

    //! Whether the reference is assigned
    explicit operator bool() const { return static_cast<bool>(ptr); }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the first element of a pinned host vector.
 *
 * The vector must not be resized while the asynchronous write is pending.
 */
template<class T>
AsyncResultRef<T>::AsyncResultRef(std::vector<T, PinnedAllocator<T>>& vec)
    : ptr{vec.data()}
{
    CELER_EXPECT(!vec.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

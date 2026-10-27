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
class AsyncResultRef
{
  public:
    //! Construct with a null reference
    AsyncResultRef() = default;

    // Construct from a pointer to pinned host memory
    inline explicit AsyncResultRef(ObserverPtr<T, MemSpace::host> dst);

    // Construct from the first element of a pinned host vector
    inline explicit AsyncResultRef(std::vector<T, PinnedAllocator<T>>& vec);

    //! Host destination of the asynchronous write
    ObserverPtr<T, MemSpace::host> ptr() const { return ptr_; }

    //! Whether the reference is assigned
    explicit operator bool() const { return static_cast<bool>(ptr_); }

  private:
    ObserverPtr<T, MemSpace::host> ptr_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a pointer to pinned host memory.
 *
 * The pointed-to memory must remain valid while the asynchronous write is
 * pending.
 */
template<class T>
AsyncResultRef<T>::AsyncResultRef(ObserverPtr<T, MemSpace::host> dst)
    : ptr_{dst}
{
    CELER_EXPECT(dst);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from the first element of a pinned host vector.
 *
 * The vector must not be resized while the asynchronous write is pending.
 */
template<class T>
AsyncResultRef<T>::AsyncResultRef(std::vector<T, PinnedAllocator<T>>& vec)
    : ptr_{vec.data()}
{
    CELER_EXPECT(!vec.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

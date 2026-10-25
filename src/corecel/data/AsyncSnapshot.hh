//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncSnapshot.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

#include "AsyncResultRef.hh"
#include "Copier.hh"
#include "ObserverPtr.hh"
#include "PinnedAllocator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enqueue a copy of a single value into pinned host memory.
 *
 * A copy from device memory is enqueued on the given stream and completes
 * asynchronously: the destination must not be read until the stream is
 * synchronized (directly or through a later event). A copy from host memory
 * completes immediately.
 */
template<class T, MemSpace M>
void copy_to_host_async(
    ObserverPtr<T const, M> src, AsyncResultRef<T> dst, StreamId stream)
{
    CELER_EXPECT(src);
    CELER_EXPECT(dst);

    auto* dst_ptr = static_cast<T*>(dst.ptr);
    auto const* src_ptr = static_cast<T const*>(src);
    if constexpr (M == MemSpace::host)
    {
        *dst_ptr = *src_ptr;
    }
    else
    {
        CELER_EXPECT(stream);
        Copier<T, MemSpace::host> copy{{dst_ptr, 1}, stream};
        copy(M, {src_ptr, 1});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Host copy of a single value that is updated asynchronously from a stream.
 *
 * The value is stored in pinned host memory so that a device-to-host copy
 * enqueued with \c copy_async does not block the host. The copied value
 * \b must not be read until the stream has completed the copy, either by
 * synchronizing the stream or an event recorded after the copy. Code that
 * enqueues its own write can be given the destination with \c ref .
 *
 * \code
    AsyncSnapshot<Counters> snapshot;
    snapshot.copy_async(device_counters, stream_id);
    done.record(device().stream(stream_id));
    // ... enqueue more work ...
    done.sync();
    use(snapshot.value());
   \endcode
 */
template<class T>
class AsyncSnapshot
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "Snapshot value must be trivially copyable");

  public:
    // Construct with an initial host value
    explicit inline AsyncSnapshot(T const& value = {});

    // Enqueue a copy of a single value from its memory space
    template<class U, MemSpace M>
    inline void copy_async(ObserverPtr<U, M> src, StreamId stream);

    //! Destination for an asynchronous write
    AsyncResultRef<T> ref() { return AsyncResultRef<T>{storage_}; }

    // Overwrite the host value
    inline void assign(T const& value);

    //! Access the most recently copied value
    T const& value() const { return storage_.front(); }

  private:
    std::vector<T, PinnedAllocator<T>> storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with an initial host value.
 */
template<class T>
AsyncSnapshot<T>::AsyncSnapshot(T const& value) : storage_(1, value)
{
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue a copy of a single value from its memory space.
 *
 * The memory space of the source is part of its type, so a device value
 * cannot be read as host memory. A copy from device memory is enqueued on the
 * given stream and completes asynchronously. A copy from host memory
 * completes immediately.
 */
template<class T>
template<class U, MemSpace M>
void AsyncSnapshot<T>::copy_async(ObserverPtr<U, M> src, StreamId stream)
{
    static_assert(std::is_same_v<std::remove_const_t<U>, T>,
                  "Source type must match the snapshot type");
    copy_to_host_async(ObserverPtr<T const, M>{src}, this->ref(), stream);
}

//---------------------------------------------------------------------------//
/*!
 * Overwrite the host value.
 *
 * The caller must ensure no asynchronous copy into the snapshot is pending.
 */
template<class T>
void AsyncSnapshot<T>::assign(T const& value)
{
    storage_.front() = value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

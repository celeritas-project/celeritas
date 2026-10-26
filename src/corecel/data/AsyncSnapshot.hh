//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncSnapshot.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <new>
#include <type_traits>

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
 * Host copy of a single value that is updated asynchronously from a stream.
 *
 * The value is stored in pinned host memory so that a device-to-host copy
 * enqueued with \c load_async does not block the host. The loaded value
 * \b must not be read until the stream has completed the copy, either by
 * synchronizing the stream or an event recorded after the copy. Code that
 * enqueues its own write can be given the destination with \c ref .
 *
 * \code
    AsyncSnapshot<Counters> snapshot;
    snapshot.load_async(device_counters, stream_id);
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

    // Load a single value from host memory
    inline void load_async(ObserverPtr<T const, MemSpace::host> src, StreamId);

    // Enqueue a load of a single value from device memory
    inline void load_async(ObserverPtr<T const, MemSpace::device> src,
                           StreamId stream);

    //! Destination for an asynchronous write
    AsyncResultRef<T> ref()
    {
        return AsyncResultRef<T>{make_observer(storage_.get())};
    }

    // Overwrite the host value
    inline void assign(T const& value);

    //! Access the most recently copied value
    T const& value() const { return *storage_; }

  private:
    struct PinnedDeleter
    {
        void operator()(T* ptr) const noexcept
        {
            PinnedAllocator<T>{}.deallocate(ptr, 1);
        }
    };

    std::unique_ptr<T, PinnedDeleter> storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with an initial host value.
 */
template<class T>
AsyncSnapshot<T>::AsyncSnapshot(T const& value)
    : storage_{new (PinnedAllocator<T>{}.allocate(1)) T(value)}
{
    CELER_ENSURE(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Load a single value from host memory.
 *
 * The load completes immediately. The stream is accepted so that code
 * templated on the memory space can call either overload.
 */
template<class T>
void AsyncSnapshot<T>::load_async(ObserverPtr<T const, MemSpace::host> src,
                                  StreamId)
{
    CELER_EXPECT(src);
    *storage_ = *static_cast<T const*>(src);
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue a load of a single value from device memory.
 *
 * The copy is enqueued on the given stream and completes asynchronously.
 */
template<class T>
void AsyncSnapshot<T>::load_async(ObserverPtr<T const, MemSpace::device> src,
                                  StreamId stream)
{
    CELER_EXPECT(src);
    CELER_EXPECT(stream);
    Copier<T, MemSpace::host> copy{{storage_.get(), 1}, stream};
    copy(MemSpace::device, {static_cast<T const*>(src), 1});
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
    *storage_ = value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

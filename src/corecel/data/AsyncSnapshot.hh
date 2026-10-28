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
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceEvent.hh"
#include "corecel/sys/Stream.hh"

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
 * enqueued with \c load_async does not block the host. Each device load
 * records an event on its stream after the copy: \c ready queries it without
 * blocking and \c sync waits for it. Because the copy is ordered after the
 * work previously enqueued on that stream, completion of the load also
 * implies completion of that work. A load from host memory completes
 * immediately.
 *
 * The loaded value \b must not be read or overwritten until the load has
 * completed, which is checked when \c CELERITAS_DEBUG is enabled.
 *
 * \code
    AsyncSnapshot<Counters> snapshot;
    snapshot.load_async(device_counters, stream_id);
    // ... enqueue more work or do host work ...
    use(snapshot.wait());
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

    //! Whether the most recent load has completed
    bool ready() const { return loaded_.ready(); }

    //! Block the host until the most recent load has completed
    void sync() const { loaded_.sync(); }

    // Overwrite the host value
    inline void assign(T const& value);

    // Access the most recently loaded value
    inline T const& value() const;

    // Wait for the most recent load and access its value
    inline T const& wait() const;

  private:
    struct PinnedDeleter
    {
        void operator()(T* ptr) const noexcept
        {
            PinnedAllocator<T>{}.deallocate(ptr, 1);
        }
    };

    std::unique_ptr<T, PinnedDeleter> storage_;
    DeviceEvent loaded_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with an initial host value.
 *
 * If a device is active, the storage is pinned and the completion event is
 * created. A snapshot that loads from device memory must therefore be
 * constructed after the device is activated.
 */
template<class T>
AsyncSnapshot<T>::AsyncSnapshot(T const& value)
    : storage_{new (PinnedAllocator<T>{}.allocate(1)) T(value)}
    , loaded_{celeritas::device()}
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
    CELER_EXPECT(this->ready());
    *storage_ = *static_cast<T const*>(src);
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue a load of a single value from device memory.
 *
 * The copy is enqueued on the given stream, followed by the completion event.
 */
template<class T>
void AsyncSnapshot<T>::load_async(ObserverPtr<T const, MemSpace::device> src,
                                  StreamId stream)
{
    CELER_EXPECT(src);
    CELER_EXPECT(stream);
    CELER_EXPECT(loaded_);
    Copier<T, MemSpace::host> copy{{storage_.get(), 1}, stream};
    copy(MemSpace::device, {static_cast<T const*>(src), 1});
    loaded_.record(celeritas::device().stream(stream));
}

//---------------------------------------------------------------------------//
/*!
 * Overwrite the host value.
 *
 * The most recent load must have completed.
 */
template<class T>
void AsyncSnapshot<T>::assign(T const& value)
{
    CELER_EXPECT(this->ready());
    *storage_ = value;
}

//---------------------------------------------------------------------------//
/*!
 * Access the most recently loaded value.
 *
 * The most recent load must have completed: call \c sync or check \c ready
 * first, or use \c wait instead.
 */
template<class T>
T const& AsyncSnapshot<T>::value() const
{
    CELER_EXPECT(this->ready());
    return *storage_;
}

//---------------------------------------------------------------------------//
/*!
 * Wait for the most recent load and access its value.
 *
 * This blocks the host until the load completes.
 */
template<class T>
T const& AsyncSnapshot<T>::wait() const
{
    this->sync();
    return *storage_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

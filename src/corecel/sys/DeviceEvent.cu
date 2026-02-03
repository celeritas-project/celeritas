//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.cu
//---------------------------------------------------------------------------//
#include "DeviceEvent.hh"

#include <iostream>

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "corecel/Assert.hh"  // IWYU pragma: keep

#include "Device.hh"
#include "Stream.hh"  // IWYU pragma: keep

using EventT = CELER_DEVICE_API_SYMBOL(Event_t);

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Internal implementation holding the native CUDA/HIP event handle.
 */
struct DeviceEvent::Impl
{
    EventT event{nullptr};
    CELER_DEVICE_API_SYMBOL(Stream_t) stream { nullptr };
};

//---------------------------------------------------------------------------//
/*! Destroy the event. */
void DeviceEvent::ImplDeleter::operator()(Impl* impl) noexcept
{
    try
    {
        CELER_DEVICE_API_CALL(EventDestroy(impl->event));
        delete impl;
    }
    catch (RuntimeError const& e)
    {
        std::cerr << "Failed to destroy event: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to destroy event" << std::endl;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct a device event for the given stream ID.
 *
 * The stream pointer is obtained from \c celeritas::device() and stored
 * internally so that \c record() can be called without passing the stream.
 */
DeviceEvent::DeviceEvent(StreamId stream_id)
    : DeviceEvent(device().stream(stream_id))
{
    CELER_EXPECT(stream_id < device().num_streams());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a device event for the given stream.
 *
 * The stream pointer is stored internally so that \c record() can be called
 * without passing the stream.
 */
DeviceEvent::DeviceEvent(Stream const& stream)
{
    EventT event;
    CELER_DEVICE_API_CALL(EventCreateWithFlags(
        &event, CELER_DEVICE_API_SYMBOL(EventDisableTiming)));
    impl_.reset(new Impl{event, stream.get()});
}

//---------------------------------------------------------------------------//
/*!
 * Record this event on the stream.
 *
 * This captures the state of the stream at the point the event is recorded.
 * All operations enqueued on the stream before this call must complete before
 * the event is considered complete.
 */
void DeviceEvent::record()
{
    CELER_EXPECT(impl_);
    CELER_DEVICE_API_CALL(EventRecord(impl_->event, impl_->stream));
}

//---------------------------------------------------------------------------//
/*!
 * Query event status without blocking.
 *
 * \return true if all operations recorded before this event have completed,
 *         false if the event is still pending
 *
 * This is a non-blocking query that returns immediately. If an error occurs
 * during the query, the function will throw an exception.
 */
bool DeviceEvent::ready() const
{
    auto result = CELER_DEVICE_API_SYMBOL(EventQuery)(impl_->event);
    if (result == CELER_DEVICE_API_SYMBOL(ErrorNotReady))
    {
        return false;
    }
    else if (result == CELER_DEVICE_API_SYMBOL(Success))
    {
        return true;
    }
    // Either is missing: an error has occurred
    CELER_DEVICE_API_CALL(GetLastError());
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Wait for the event to complete.
 *
 * This blocks the calling thread until all operations recorded before this
 * event have finished executing on the device. Use this to synchronize the
 * host with device operations.
 */
void DeviceEvent::sync() const
{
    CELER_EXPECT(impl_);
    CELER_DEVICE_API_CALL(EventSynchronize(impl_->event));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

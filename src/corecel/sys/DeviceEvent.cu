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

#include "Stream.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Internal implementation holding the native CUDA/HIP event handle.
 */
struct DeviceEvent::Impl
{
    EventT event{nullptr};
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
 * Construct a device event with timing disabled.
 */
DeviceEvent::DeviceEvent()
{
    EventT event;
    CELER_DEVICE_API_CALL(EventCreateWithFlags(
        &event, CELER_DEVICE_API_SYMBOL(EventDisableTiming)));
    impl_.reset(new Impl{event});
}

//---------------------------------------------------------------------------//
/*!
 * Get the native CUDA/HIP event handle.
 *
 * This provides direct access to the underlying event for advanced use cases.
 */
DeviceEvent::EventT DeviceEvent::get() const
{
    CELER_EXPECT(impl_);
    return impl_->event;
}

//---------------------------------------------------------------------------//
/*!
 * Record this event on a stream.
 *
 * This captures the state of the stream at the point the event is recorded.
 * All operations enqueued on the stream before this call must complete before
 * the event is considered complete.
 */
void DeviceEvent::record(Stream const& stream) const
{
    CELER_EXPECT(impl_);
    CELER_DEVICE_API_CALL(EventRecord(impl_->event, stream.get()));
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

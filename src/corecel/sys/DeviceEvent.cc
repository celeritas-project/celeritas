//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.cc
//---------------------------------------------------------------------------//
#include "DeviceEvent.hh"

#include <iostream>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"  // IWYU pragma: keep

#include "Device.hh"
#include "Stream.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Internal implementation holding the native CUDA/HIP event handle.
 */
struct DeviceEvent::Impl
{
#if CELER_USE_DEVICE
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
#else
    using StreamT = nullptr_t;
#endif

    EventT event{nullptr};
    StreamT stream{nullptr};
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
 *
 * \pre A device must be active and the stream ID must be valid.
 */
DeviceEvent::DeviceEvent(StreamId stream_id)
    : DeviceEvent(device().stream(stream_id))
{
    CELER_VALIDATE(celeritas::device(),
                   << "cannot construct event: device is not configured");
    CELER_VALIDATE(stream_id < device().num_streams(),
                   << "invalid stream ID " << stream_id.unchecked_get()
                   << " (only " << device().num_streams()
                   << " streams available)");
}

//---------------------------------------------------------------------------//
/*!
 * Construct a device event for the given stream.
 *
 * The stream pointer is stored internally so that \c record() can be called
 * without passing the stream.
 *
 * \pre The stream must be valid (not null or moved-from).
 */
DeviceEvent::DeviceEvent(Stream const& stream)
{
    CELER_VALIDATE(stream, << "cannot construct event: stream is not valid");
    EventT event;
    CELER_DEVICE_API_CALL(EventCreateWithFlags(
        &event, CELER_DEVICE_API_SYMBOL(EventDisableTiming)));
    impl_.reset(new Impl{event, stream.get()});
}

//---------------------------------------------------------------------------//
/*!
 * Construct a null device event.
 */
DeviceEvent::DeviceEvent(std::nullptr_t)
{
    // Do not create an event; leave impl_ as nullptr
}

//---------------------------------------------------------------------------//
/*!
 * Whether the event is valid (not null or moved-from).
 */
DeviceEvent::operator bool() const
{
    return impl_ != nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get the native CUDA/HIP event handle.
 *
 * This provides direct access to the underlying event for advanced use cases.
 */
auto DeviceEvent::get() const -> EventT
{
    if (!impl_)
        return nullptr;

    return impl_->event;
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
    if (!*this)
        return;

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
    if (!*this)
        return true;

#if CELER_USE_DEVICE
    auto result = CELER_DEVICE_API_SYMBOL(EventQuery)(impl_->event);
    if (result == CELER_DEVICE_API_SYMBOL(ErrorNotReady))
    {
        return false;
    }
    else if (result == CELER_DEVICE_API_SYMBOL(Success))
    {
        return true;
    }
    CELER_RUNTIME_THROW(CELER_DEVICE_PLATFORM_UPPER_STR,
                        CELER_DEVICE_API_SYMBOL(GetErrorString)(result),
                        "EventQuery");
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Wait for the event to complete.
 *
 * This blocks the calling thread until all operations recorded before this
 * event have finished executing on the device. Use this to synchronize the
 * host with device operations.
 *
 * If the event is uninitialized
 */
void DeviceEvent::sync() const
{
    if (!*this)
        return;

    CELER_DEVICE_API_CALL(EventSynchronize(impl_->event));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.cc
//---------------------------------------------------------------------------//
// NOTE: runtime API *must* be included before header
#include "corecel/DeviceRuntimeApi.hh"
//------------------------------------//
#include "DeviceEvent.hh"
//------------------------------------//
#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "Device.hh"
#include "Stream.hh"

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
//! Destroy the event.
void DeviceEvent::ImplDeleter::operator()(Impl* impl) noexcept
{
    if constexpr (CELER_USE_DEVICE)
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
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct a device event.
 */
DeviceEvent::DeviceEvent(Device const& d)
{
    if (d)
    {
        EventT event;
        CELER_DEVICE_API_CALL(EventCreateWithFlags(
            &event, CELER_DEVICE_API_SYMBOL(EventDisableTiming)));
        impl_.reset(new Impl{event});
    }
    CELER_ENSURE(static_cast<bool>(*this) == static_cast<bool>(d));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a null device event.
 */
DeviceEvent::DeviceEvent(std::nullptr_t)
{
    CELER_ENSURE(!*this);
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
void DeviceEvent::record(Stream const& s)
{
    CELER_EXPECT(*this);
    CELER_DEVICE_API_CALL(EventRecord(impl_->event, s.get()));
#if !CELER_USE_DEVICE
    CELER_DISCARD(s);
#endif
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
 * If the event is null, this is a no-op.
 */
void DeviceEvent::sync() const
{
    if (!*this)
        return;

    CELER_DEVICE_API_CALL(EventSynchronize(impl_->event));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.cu
//---------------------------------------------------------------------------//
#include "DeviceEvent.hh"

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "corecel/Assert.hh"  // IWYU pragma: keep

#include "Stream.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
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
        std::cerr << "Failed to destroy stream: " << e.what() << std::endl;
    }
    catch (...)
    {
        // Destructors must not throw
    }
}

//---------------------------------------------------------------------------//
DeviceEvent::DeviceEvent()
{
    EventT event;
    CELER_DEVICE_API_CALL(EventCreateWithFlags(
        &event, CELER_DEVICE_API_SYMBOL(EventDisableTiming)));
    impl_.reset(new Impl{event});
}

//---------------------------------------------------------------------------//
DeviceEvent::EventT DeviceEvent::get() const
{
    CELER_EXPECT(impl_);
    return impl_->event;
}

//---------------------------------------------------------------------------//
void DeviceEvent::record(Stream const& stream) const
{
    CELER_EXPECT(impl_);
    CELER_DEVICE_API_CALL(EventRecord(impl_->event, stream.get()));
}

//---------------------------------------------------------------------------//
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
void DeviceEvent::sync() const
{
    CELER_EXPECT(impl_);
    CELER_DEVICE_API_CALL(EventSynchronize(impl_->event));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

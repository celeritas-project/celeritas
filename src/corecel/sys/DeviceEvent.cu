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

#if CELER_USE_DEVICE
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
DeviceEvent::Status DeviceEvent::status() const
{
    CELER_EXPECT(impl_);

    auto result = CELER_DEVICE_API_SYMBOL(EventQuery)(impl_->event);
    if (result == CELER_DEVICE_API_SYMBOL(ErrorNotReady))
    {
        return Status::pending;
    }
    CELER_DEVICE_API_CALL(EventQuery(impl_->event));
    return Status::ready;
}

//---------------------------------------------------------------------------//
bool DeviceEvent::ready() const
{
    return this->status() == Status::ready;
}

//---------------------------------------------------------------------------//
void DeviceEvent::sync() const
{
    CELER_EXPECT(impl_);
    CELER_DEVICE_API_CALL(EventSynchronize(impl_->event));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
#endif  // CELER_USE_DEVICE

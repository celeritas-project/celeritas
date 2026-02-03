//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"  // IWYU pragma: keep

#include "corecel/Assert.hh"  // IWYU pragma: keep
#include "corecel/Macros.hh"

#if CELER_DEVICE_SOURCE
#    include "corecel/DeviceRuntimeApi.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
class Stream;

//---------------------------------------------------------------------------//
/*! Minimal wrapper around a CUDA/HIP event. */
class DeviceEvent
{
  public:
#ifdef CELER_DEVICE_RUNTIME_INCLUDED
    using EventT = CELER_DEVICE_API_SYMBOL(Event_t);
#else
    using MissingDeviceRuntime = void;
#endif

  public:
    DeviceEvent();
    CELER_DEFAULT_MOVE_DELETE_COPY(DeviceEvent);
    ~DeviceEvent() = default;

#ifdef CELER_DEVICE_RUNTIME_INCLUDED
    EventT get() const;
#else
    MissingDeviceRuntime get() const {}
#endif

    // Record this event on a stream
    void record(Stream const& stream) const;

    // Query event status
    bool ready() const;

    // Wait for the event to complete
    void sync() const;

  private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl*) noexcept;
    };

    std::unique_ptr<Impl, ImplDeleter> impl_{};
};

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline DeviceEvent::DeviceEvent() = default;

inline void DeviceEvent::record(Stream const&) const {}

inline bool DeviceEvent::ready() const
{
    return true;
}

inline void DeviceEvent::sync() const {}

inline void DeviceEvent::ImplDeleter::operator()(Impl*) noexcept {}

#    ifdef CELER_DEVICE_RUNTIME_INCLUDED
inline DeviceEvent::EventT DeviceEvent::get() const
{
    CELER_ASSERT_UNREACHABLE();
}
#    endif
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas

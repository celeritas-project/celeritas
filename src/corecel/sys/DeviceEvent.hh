//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Assert.hh"  // IWYU pragma: keep
#include "corecel/Macros.hh"
#include "corecel/sys/ThreadId.hh"

#if CELER_DEVICE_SOURCE
#    include "corecel/DeviceRuntimeApi.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
class Device;
class Stream;

//---------------------------------------------------------------------------//
/*!
 * Minimal wrapper around a CUDA/HIP event for synchronization.
 *
 * Events provide a mechanism for querying the status of asynchronous
 * operations on GPU streams and synchronizing between host and device, and
 * synchronizing between streams.
 *
 * \par States
 * - \b Constructed: when built with an active device, the instance
 *   evaluates to \c true and manages an event object.
 * - \b Null: when constructed with a nullptr, or when \c moved from, the class
 *   instance is \c false. It does not manage an event.
 *   The \c sync function is a null-op (the event is always
 *   \c ready ), and \c record cannot be called.
 *
 * If no device is enabled (or Celeritas is compiled without CUDA/HIP support),
 * only the null state is available.
 *
 * \par Example
 * \code
  // Setup:
  DeviceEvent my_kernel(celeritas::device());
  assert(my_kernel.ready());
  // Later...
  launch_kernel_async(state.stream());
  my_kernel.record(state.stream());
  // Then do CPU work until the kernel or CPU is done
  while (!cpu_work_done() && !my_kernel.ready())
  {
      cpu_work();
  }
 * \endcode
 * Use \c my_kernel.sync() before the kernel launch to wait on the previous
 * kernel launch before going again.
 */
class DeviceEvent
{
  public:
#if !CELER_USE_DEVICE
    //! Event implementation is unavailable
    using EventT = std::nullptr_t;
#elif !defined(CELER_DEVICE_RUNTIME_INCLUDED)
    //! Sentinel type to indicate compilation error: include runtime downstream
    using MissingDeviceRuntime = void;
#else
    //! Actual CUDA/HIP stream opaque pointer
    using EventT = CELER_DEVICE_API_SYMBOL(Event_t);
#endif

  public:
    // Construct with a device context
    explicit DeviceEvent(Device const& d);
    // Construct a null event
    DeviceEvent(std::nullptr_t);
    CELER_DEFAULT_MOVE_DELETE_COPY(DeviceEvent);
    ~DeviceEvent() = default;

    // Whether the event is valid (not null or moved-from)
    inline explicit operator bool() const;

#if defined(CELER_DEVICE_RUNTIME_INCLUDED) || !CELER_USE_DEVICE
    EventT get() const;
#else
    MissingDeviceRuntime get() const {}
#endif

    // Record this event on the stream
    void record(Stream const& s);

    // Query event status
    bool ready() const;

    // Block the host until the recorded event is complete
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
/*!
 * Whether the event is valid (not null or moved-from).
 */
DeviceEvent::operator bool() const
{
    return static_cast<bool>(impl_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

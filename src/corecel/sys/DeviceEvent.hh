//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"  // IWYU pragma: keep
#include "corecel/Macros.hh"
#include "corecel/sys/ThreadId.hh"

#if CELER_USE_DEVICE
#    include "corecel/DeviceRuntimeApi.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
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
 * - \b Constructed: when build with a device stream object, the instance
 *   evaluates to \c true and forwards operations to device APIs.
 * - \b Null: when constructed with a nullptr, or when \c moved from, the class
 *   instance is \c false. It does not manage an event nor does it associate
 *   with a stream. The \c sync  and \c record functions are null-op, the event
 *   is always "ready", and the host kernel launch is instantaneous.
 *
 * If no device is enabled (or Celeritas is compiled without CUDA/HIP support),
 * only the nullptr constructor is allowed.
 *
 * \par Example:
 * \code
  // Setup:
  DeviceEvent my_kernel(state.stream_id());
  assert(my_kernel.ready());
  // Later...
  launch_kernel_async(state);
  my_kernel.record();
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
    using EventT = nullptr_t;
#elif !CELER_DEVICE_RUNTIME_INCLUDED
    //! Sentinel type to indicate compilation error: include runtime downstream
    using MissingDeviceRuntime = void;
#else
    //! Actual CUDA/HIP stream opaque pointer
    using EventT = CELER_DEVICE_API_SYMBOL(Event_t);
#endif

  public:
    // Construct with stream or stream ID
    explicit DeviceEvent(StreamId stream_id);
    explicit DeviceEvent(Stream const& stream);
    // Construct a null event
    DeviceEvent(std::nullptr_t);
    CELER_DEFAULT_MOVE_DELETE_COPY(DeviceEvent);
    ~DeviceEvent() = default;

    // Whether the event is valid (not null or moved-from)
    explicit operator bool() const;

#if defined(CELER_DEVICE_RUNTIME_INCLUDED) || !CELER_USE_DEVICE
    EventT get() const;
#else
    MissingDeviceRuntime get() const {}
#endif

    // Record this event on the stream
    void record();

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

// Block stream execution until the event is complete
void stream_wait_event(Stream& s, DeviceEvent const& e);

//---------------------------------------------------------------------------//
}  // namespace celeritas

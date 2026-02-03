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
#include "corecel/sys/ThreadId.hh"

#if CELER_DEVICE_SOURCE
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
 * operations on GPU streams and synchronizing between host and device. This
 * class creates events with timing disabled for minimal overhead.
 *
 * When CUDA/HIP is unavailable, this class provides a no-op implementation.
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
#ifdef CELER_DEVICE_RUNTIME_INCLUDED
    using EventT = CELER_DEVICE_API_SYMBOL(Event_t);
#else
    using MissingDeviceRuntime = void;
#endif

  public:
    // Construct with stream or stream ID
    explicit DeviceEvent(StreamId stream_id);
    explicit DeviceEvent(Stream const& stream);

#ifdef CELER_DEVICE_RUNTIME_INCLUDED
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

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline DeviceEvent::DeviceEvent(StreamId) {}

inline DeviceEvent::DeviceEvent(Stream const&) {}

inline void DeviceEvent::record() {}

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

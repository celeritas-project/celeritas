//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/CounterAlgorithms.cu
//---------------------------------------------------------------------------//
#include "CounterAlgorithms.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"

#include "detail/CounterExecutors.hh"

namespace celeritas
{
namespace
{
using CSCDeviceRef = CoreStateCounterRef<MemSpace::device>;
}  // namespace
//---------------------------------------------------------------------------//
void reset_counters(CSCDeviceRef const& counters, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::ResetCountersExecutor execute_thread{counters.data()};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "reset-counters");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
void add_pending(
    CSCDeviceRef const& counters, size_type count, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::AddPendingExecutor<size_type> execute_thread{counters.data(),
                                                         count};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "add-pending");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
void add_pending(CSCDeviceRef const& counters,
                 ObserverPtr<size_type, MemSpace::device> count,
                 StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::AddPendingExecutor<size_type*> execute_thread{
        counters.data(), static_cast<size_type*>(count)};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "add-pending");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
void add_primaries(
    CSCDeviceRef const& counters, size_type count, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::AddPrimariesCountExecutor execute_thread{counters.data(), count};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "add-primary");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
void update_active(
    CSCDeviceRef const& counters, size_type state_size, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::UpdateActiveExecutor execute_thread{counters.data(), state_size};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-active");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
void update_alive(
    CSCDeviceRef const& counters, size_type state_size, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::UpdateAliveExecutor execute_thread{counters.data(), state_size};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-alive");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
void update_secondaries(
    CSCDeviceRef const& counters,
    ObserverPtr<size_type, MemSpace::device> num_secondaries,
    size_type state_size,
    StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    detail::UpdateSecondariesCountExecutor execute_thread{
        counters.data(), num_secondaries, state_size};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "update-secondaries");
    launch_kernel(1, stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

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
void initialize_counters(
    CSCDeviceRef const& counters, size_type num_pending, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel(
        "reset-counters",
        stream_id,
        detail::InitializeCountersExecutor{counters.data(), num_pending});
}

//---------------------------------------------------------------------------//
void clear_tallies(CSCDeviceRef const& counters, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel("clear-tallies",
                         stream_id,
                         detail::ClearTalliesExecutor{counters.data()});
}

void add_pending(
    CSCDeviceRef const& counters, size_type count, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel(
        "add-pending",
        stream_id,
        detail::AddPendingExecutor<size_type>{counters.data(), count});
}

//---------------------------------------------------------------------------//
void add_pending(CSCDeviceRef const& counters,
                 ObserverPtr<size_type, MemSpace::device> count,
                 StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel("add-pending",
                         stream_id,
                         detail::AddPendingExecutor<size_type*>{
                             counters.data(), static_cast<size_type*>(count)});
}

//---------------------------------------------------------------------------//
void add_primaries(
    CSCDeviceRef const& counters, size_type count, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel(
        "add-primary",
        stream_id,
        detail::AddPrimariesCountExecutor{counters.data(), count});
}

//---------------------------------------------------------------------------//
void update_active(
    CSCDeviceRef const& counters, size_type state_size, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel(
        "update-active",
        stream_id,
        detail::UpdateActiveExecutor{counters.data(), state_size});
}

//---------------------------------------------------------------------------//
void update_alive(
    CSCDeviceRef const& counters, size_type state_size, StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel(
        "update-alive",
        stream_id,
        detail::UpdateAliveExecutor{counters.data(), state_size});
}

//---------------------------------------------------------------------------//
void update_secondaries(
    CSCDeviceRef const& counters,
    ObserverPtr<size_type, MemSpace::device> num_secondaries,
    size_type state_size,
    StreamId stream_id)
{
    CELER_EXPECT(counters.size() == 1);
    launch_single_kernel("update-secondaries",
                         stream_id,
                         detail::UpdateSecondariesCountExecutor{
                             counters.data(), num_secondaries, state_size});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

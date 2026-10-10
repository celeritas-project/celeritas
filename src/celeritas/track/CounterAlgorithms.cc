//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/CounterAlgorithms.cc
//---------------------------------------------------------------------------//
#include "CounterAlgorithms.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.hh"

#include "detail/CounterExecutors.hh"

namespace celeritas
{
namespace
{
using CSCHostRef = CoreStateCounterRef<MemSpace::host>;
using CSCDeviceRef = CoreStateCounterRef<MemSpace::device>;
}  // namespace
//---------------------------------------------------------------------------//
void initialize_counters(
    CSCHostRef const& counters, size_type num_pending, StreamId)
{
    CELER_EXPECT(counters.size() == 1);
    launch_kernel(
        1, detail::InitializeCountersExecutor{counters.data(), num_pending});
}

//---------------------------------------------------------------------------//
void add_pending(CSCHostRef const& counters, size_type count, StreamId)
{
    CELER_EXPECT(counters.size() == 1);
    launch_kernel(
        1, detail::AddPendingExecutor<size_type>{counters.data(), count});
}

//---------------------------------------------------------------------------//
void add_pending(CSCHostRef const& counters,
                 ObserverPtr<size_type, MemSpace::host> count,
                 StreamId stream_id)
{
    add_pending(counters, *count, stream_id);
}

//---------------------------------------------------------------------------//
void add_primaries(CSCHostRef const& counters, size_type count, StreamId)
{
    CELER_EXPECT(count > 0);
    CELER_EXPECT(counters.size() == 1);
    launch_kernel(1, detail::AddPrimariesCountExecutor{counters.data(), count});
}

//---------------------------------------------------------------------------//
void update_active(CSCHostRef const& counters, size_type state_size, StreamId)
{
    CELER_EXPECT(counters.size() == 1);
    launch_kernel(1, detail::UpdateActiveExecutor{counters.data(), state_size});
}

//---------------------------------------------------------------------------//
void update_alive(CSCHostRef const& counters, size_type state_size, StreamId)
{
    CELER_EXPECT(counters.size() == 1);
    launch_kernel(1, detail::UpdateAliveExecutor{counters.data(), state_size});
}

//---------------------------------------------------------------------------//
void update_secondaries(CSCHostRef const& counters,
                        ObserverPtr<size_type, MemSpace::host> num_secondaries,
                        size_type state_size,
                        StreamId)
{
    CELER_EXPECT(counters.size() == 1);
    launch_kernel(1,
                  detail::UpdateSecondariesCountExecutor{
                      counters.data(), num_secondaries, state_size});
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void initialize_counters(CSCDeviceRef const&, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void add_pending(CSCDeviceRef const&, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void add_pending(
    CSCDeviceRef const&, ObserverPtr<size_type, MemSpace::device>, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void add_primaries(CSCDeviceRef const&, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void update_active(CSCDeviceRef const&, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void update_alive(CSCDeviceRef const&, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void update_secondaries(CSCDeviceRef const&,
                        ObserverPtr<size_type, MemSpace::device>,
                        size_type,
                        StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

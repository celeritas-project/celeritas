//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorCounterAlgorithms.cc
//---------------------------------------------------------------------------//
#include "GeneratorCounterAlgorithms.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.hh"

#include "GeneratorCounterExecutors.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
namespace
{
using HostGenPtr = GenCountersObserver<MemSpace::host>;
using DeviceGenPtr = GenCountersObserver<MemSpace::device>;
}  // namespace

//---------------------------------------------------------------------------//
void assign_counters(
    HostGenPtr counters, GeneratorCounters<> const& value, StreamId)
{
    CELER_EXPECT(counters);
    launch_kernel(1, AssignGeneratorCountersExecutor{counters, value});
}

//---------------------------------------------------------------------------//
void clear_accum(HostGenPtr counters, StreamId)
{
    CELER_EXPECT(counters);
    launch_kernel(1, ClearAccumExecutor{counters});
}

//---------------------------------------------------------------------------//
void clear_generated(HostGenPtr counters, StreamId)
{
    CELER_EXPECT(counters);
    launch_kernel(1, ClearGeneratedExecutor{counters});
}

//---------------------------------------------------------------------------//
void accumulate_buffer(HostGenPtr counters, StreamId)
{
    CELER_EXPECT(counters);
    launch_kernel(1, AccumulateBufferExecutor{counters});
}

//---------------------------------------------------------------------------//
void set_buffer_size(HostGenPtr counters,
                     ObserverPtr<size_type const, MemSpace::host> num_selected,
                     StreamId)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(num_selected);
    launch_kernel(1, SetBufferSizeExecutor{counters, num_selected});
}

//---------------------------------------------------------------------------//
void update_pending(HostGenPtr counters,
                    ObserverPtr<size_type const, MemSpace::host> total,
                    StreamId)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(total);
    launch_kernel(1, UpdatePendingExecutor{counters, total});
}

//---------------------------------------------------------------------------//
void update_counters(CoreStateCounterRef<MemSpace::host> const& core_counters,
                     HostGenPtr counters,
                     size_type state_size,
                     StreamId)
{
    CELER_EXPECT(core_counters.size() == 1);
    CELER_EXPECT(counters);
    launch_kernel(1,
                  UpdateGeneratorCountersExecutor{
                      core_counters.data(), counters, state_size});
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void assign_counters(DeviceGenPtr, GeneratorCounters<> const&, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void clear_accum(DeviceGenPtr, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void clear_generated(DeviceGenPtr, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void accumulate_buffer(DeviceGenPtr, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void set_buffer_size(
    DeviceGenPtr, ObserverPtr<size_type const, MemSpace::device>, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void update_pending(
    DeviceGenPtr, ObserverPtr<size_type const, MemSpace::device>, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void update_counters(CoreStateCounterRef<MemSpace::device> const&,
                     DeviceGenPtr,
                     size_type,
                     StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

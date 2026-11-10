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
    launch_single_kernel(AssignGeneratorCountersExecutor{counters, value});
}

//---------------------------------------------------------------------------//
void clear_accum(HostGenPtr counters, StreamId)
{
    CELER_EXPECT(counters);
    launch_single_kernel(ClearAccumExecutor{counters});
}

//---------------------------------------------------------------------------//
void clear_generated(HostGenPtr counters, StreamId)
{
    CELER_EXPECT(counters);
    launch_single_kernel(ClearGeneratedExecutor{counters});
}

//---------------------------------------------------------------------------//
void accumulate_buffer(HostGenPtr counters, StreamId)
{
    CELER_EXPECT(counters);
    launch_single_kernel(AccumulateBufferExecutor{counters});
}

//---------------------------------------------------------------------------//
void set_buffer_size(HostGenPtr counters,
                     ObserverPtr<size_type const, MemSpace::host> num_selected,
                     StreamId)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(num_selected);
    launch_single_kernel(SetBufferSizeExecutor{counters, num_selected});
}

//---------------------------------------------------------------------------//
void add_buffer_size(HostGenPtr counters,
                     ObserverPtr<size_type const, MemSpace::host> count,
                     StreamId)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(count);
    launch_single_kernel(AddBufferSizeExecutor{counters, count});
}

//---------------------------------------------------------------------------//
void update_pending(HostGenPtr counters,
                    ObserverPtr<size_type const, MemSpace::host> total,
                    StreamId)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(total);
    launch_single_kernel(UpdatePendingExecutor{counters, total});
}

//---------------------------------------------------------------------------//
void update_counters(CoreStateCounterRef<MemSpace::host> const& core_counters,
                     HostGenPtr counters,
                     size_type state_size,
                     StreamId)
{
    CELER_EXPECT(core_counters.size() == 1);
    CELER_EXPECT(counters);
    launch_single_kernel(UpdateGeneratorCountersExecutor{
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

void add_buffer_size(
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

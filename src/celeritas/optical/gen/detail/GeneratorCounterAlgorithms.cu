//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorCounterAlgorithms.cu
//---------------------------------------------------------------------------//
#include "GeneratorCounterAlgorithms.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.device.hh"

#include "GeneratorCounterExecutors.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
namespace
{
using DeviceGenPtr = GenCountersObserver<MemSpace::device>;
}  // namespace

//---------------------------------------------------------------------------//
void assign_counters(DeviceGenPtr counters,
                     GeneratorCounters<> const& value,
                     StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single_kernel("assign-generator-counters",
                         stream_id,
                         AssignGeneratorCountersExecutor{counters, value});
}

//---------------------------------------------------------------------------//
void clear_accum(DeviceGenPtr counters, StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single_kernel(
        "clear-generator-accum", stream_id, ClearAccumExecutor{counters});
}

//---------------------------------------------------------------------------//
void clear_generated(DeviceGenPtr counters, StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single_kernel("clear-generator-generated",
                         stream_id,
                         ClearGeneratedExecutor{counters});
}

//---------------------------------------------------------------------------//
void accumulate_buffer(DeviceGenPtr counters, StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single_kernel("accumulate-generator-buffer",
                         stream_id,
                         AccumulateBufferExecutor{counters});
}

//---------------------------------------------------------------------------//
void set_buffer_size(
    DeviceGenPtr counters,
    ObserverPtr<size_type const, MemSpace::device> num_selected,
    StreamId stream_id)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(num_selected);
    launch_single_kernel("set-generator-buffer-size",
                         stream_id,
                         SetBufferSizeExecutor{counters, num_selected});
}

//---------------------------------------------------------------------------//
void update_pending(DeviceGenPtr counters,
                    ObserverPtr<size_type const, MemSpace::device> total,
                    StreamId stream_id)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(total);
    launch_single_kernel("update-generator-pending",
                         stream_id,
                         UpdatePendingExecutor{counters, total});
}

//---------------------------------------------------------------------------//
void update_counters(CoreStateCounterRef<MemSpace::device> const& core_counters,
                     DeviceGenPtr counters,
                     size_type state_size,
                     StreamId stream_id)
{
    CELER_EXPECT(core_counters.size() == 1);
    CELER_EXPECT(counters);
    launch_single_kernel("update-generator-counters",
                         stream_id,
                         UpdateGeneratorCountersExecutor{
                             core_counters.data(), counters, state_size});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

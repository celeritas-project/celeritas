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

//---------------------------------------------------------------------------//
//! Launch a single-thread executor on a stream
template<class F>
void launch_single(char const* label, F const& execute_thread, StreamId stream)
{
    static KernelLauncher<F> const launch_kernel(label);
    launch_kernel(1, stream, execute_thread);
}
}  // namespace

//---------------------------------------------------------------------------//
void assign_counters(DeviceGenPtr counters,
                     GeneratorCounters<> const& value,
                     StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single("assign-generator-counters",
                  AssignGeneratorCountersExecutor{counters, value},
                  stream_id);
}

//---------------------------------------------------------------------------//
void clear_accum(DeviceGenPtr counters, StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single(
        "clear-generator-accum", ClearAccumExecutor{counters}, stream_id);
}

//---------------------------------------------------------------------------//
void clear_generated(DeviceGenPtr counters, StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single("clear-generator-generated",
                  ClearGeneratedExecutor{counters},
                  stream_id);
}

//---------------------------------------------------------------------------//
void accumulate_buffer(DeviceGenPtr counters, StreamId stream_id)
{
    CELER_EXPECT(counters);
    launch_single("accumulate-generator-buffer",
                  AccumulateBufferExecutor{counters},
                  stream_id);
}

//---------------------------------------------------------------------------//
void set_buffer_size(
    DeviceGenPtr counters,
    ObserverPtr<size_type const, MemSpace::device> num_selected,
    StreamId stream_id)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(num_selected);
    launch_single("set-generator-buffer-size",
                  SetBufferSizeExecutor{counters, num_selected},
                  stream_id);
}

//---------------------------------------------------------------------------//
void update_pending(DeviceGenPtr counters,
                    ObserverPtr<size_type const, MemSpace::device> total,
                    StreamId stream_id)
{
    CELER_EXPECT(counters);
    CELER_EXPECT(total);
    launch_single("update-generator-pending",
                  UpdatePendingExecutor{counters, total},
                  stream_id);
}

//---------------------------------------------------------------------------//
void update_counters(CoreStateCounterRef<MemSpace::device> const& core_counters,
                     DeviceGenPtr counters,
                     size_type state_size,
                     StreamId stream_id)
{
    CELER_EXPECT(core_counters.size() == 1);
    CELER_EXPECT(counters);
    launch_single("update-generator-counters",
                  UpdateGeneratorCountersExecutor{
                      core_counters.data(), counters, state_size},
                  stream_id);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

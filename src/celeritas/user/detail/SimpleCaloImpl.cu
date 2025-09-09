//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SimpleCaloImpl.cu
//---------------------------------------------------------------------------//
#include "SimpleCaloImpl.hh"

#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.device.hh"

#include "SimpleCaloExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition on device.
 */
void simple_calo_accum(DeviceRef<StepStateData> const& step,
                       DeviceRef<SimpleCaloStateData>& calo)
{
    CELER_EXPECT(step && calo);

    SimpleCaloExecutor execute_thread{step, calo};
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "simple-calo-accum");
    launch_kernel(step.size(), step.stream_id, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

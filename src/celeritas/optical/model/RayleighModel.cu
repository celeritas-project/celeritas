//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.cu
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "RayleighExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Interact with device data.
 */
void RayleighModel::step(CoreParams const& params, CoreStateDevice& state) const
{
#if (CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_RANLUXPP)
    // Advance the random number generator
    auto execute_rng_advance
        = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                      state.ptr,
                                      this->action_id(),
                                      InteractionApplier{RNGExecutor{}});
    static ActionLauncher<decltype(execute_rng_advance)> const launch_rng_kernel(
        *this);
    launch_rng_kernel(state, execute_rng_advance);
#endif

    // Execute Rayleigh scattering
    auto execute_rayleigh
        = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                      state.ptr(),
                                      this->action_id(),
                                      InteractionApplier{RayleighExecutor{}});
    static ActionLauncher<decltype(execute_rayleigh)> const launch_kernel(
        *this);
    launch_kernel(state, execute_rayleigh);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

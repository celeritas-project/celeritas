//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/CherenkovGeneratorAction.cu
//---------------------------------------------------------------------------//
#include "CherenkovGeneratorAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "CherenkovGeneratorExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "../CherenkovParams.hh"
#include "../OffloadParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical photon initializers.
 */
void CherenkovGeneratorAction::generate(CoreParams const& core_params,
                                        CoreStateDevice& core_state) const
{
    auto& offload_state = get<OpticalOffloadState<MemSpace::native>>(
        core_state.aux(), offload_id_);
    auto& optical_state = get<optical::CoreState<MemSpace::native>>(
        core_state.aux(), optical_id_);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::CherenkovGeneratorExecutor{core_state.ptr(),
                                           material_->device_ref(),
                                           cherenkov_->device_ref(),
                                           offload_state.store.ref(),
                                           optical_state.ptr(),
                                           offload_state.buffer_size,
                                           optical_state.counters()}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

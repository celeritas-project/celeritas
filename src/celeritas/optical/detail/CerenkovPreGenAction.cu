//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovPreGenAction.cu
//---------------------------------------------------------------------------//
#include "CerenkovPreGenAction.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/OpticalPropertyParams.hh"

#include "CerenkovPreGenExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "OpticalGenStorage.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to generate optical distribution data post-step.
 */
void CerenkovPreGenAction::pre_generate(CoreParams const& core_params,
                                        CoreStateDevice& core_state) const
{
    TrackExecutor execute{core_params.ptr<MemSpace::native>(),
                          core_state.ptr(),
                          detail::CerenkovPreGenExecutor{
                              properties_->device_ref(),
                              cerenkov_->device_ref(),
                              storage_->obj.state<MemSpace::native>(
                                  core_state.stream_id(), core_state.size()),
                              storage_->size[core_state.stream_id().get()]}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

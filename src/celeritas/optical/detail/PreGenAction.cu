//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenAction.cu
//---------------------------------------------------------------------------//
#include "PreGenAction.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Thrust.device.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/OpticalPropertyParams.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "OpticalGenStorage.hh"
#include "PreGenExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
void PreGenAction::pre_generate(CoreParams const& core_params,
                                CoreStateDevice& core_state) const
{
    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::PreGenExecutor{properties_->device_ref(),
                               cerenkov_->device_ref(),
                               scintillation_->device_ref(),
                               storage_->obj.state<MemSpace::native>(
                                   core_state.stream_id(), core_state.size()),
                               storage_->size[core_state.stream_id().get()]}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(core_state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 */
size_type
PreGenAction::remove_if_invalid(ItemsRef<MemSpace::device> const& buffer,
                                size_type offset,
                                size_type size,
                                StreamId stream) const
{
    ScopedProfiling profile_this{"remove-if-invalid"};
    auto start = thrust::device_pointer_cast(buffer.data().get());
    auto stop = thrust::remove_if(thrust_execute_on(stream),
                                  start + offset,
                                  start + offset + size,
                                  IsInvalid{});
    CELER_DEVICE_CHECK_ERROR();
    return stop - start;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

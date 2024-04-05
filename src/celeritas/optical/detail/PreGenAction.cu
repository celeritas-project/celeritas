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

#include "GenStorage.hh"
#include "PreGenExecutor.hh"
#include "PreGenGatherExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
template<>
void PreGenAction<StepPoint::pre>::execute(CoreParams const& params,
                                           CoreStateDevice& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenGatherExecutor{storage_->obj.state<MemSpace::native>(
            state.stream_id(), state.size())});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 *
 * TODO: execute_impl to reduce duplicate code
 */
template<>
void PreGenAction<StepPoint::post>::execute(CoreParams const& params,
                                            CoreStateDevice& state) const
{
    StreamId stream = state.stream_id();
    auto& offsets = offsets_[stream.get()];
    auto const& gen_state
        = storage_->obj.state<MemSpace::native>(stream, state.size());
    CELER_VALIDATE(offsets.cerenkov + state.size() <= gen_state.cerenkov.size(),
                   << "insufficient capacity (" << gen_state.cerenkov.size()
                   << ") for buffered Cerenkov distribution data (total "
                      "capacity requirement of "
                   << offsets.cerenkov + state.size() << ")");
    CELER_VALIDATE(
        offsets.scintillation + state.size() <= gen_state.scintillation.size(),
        << "insufficient capacity (" << gen_state.scintillation.size()
        << ") for buffered scintillation distribution data (total "
           "capacity requirement of "
        << offsets.scintillation + state.size() << ")");

    TrackExecutor execute{params.ptr<MemSpace::native>(),
                          state.ptr(),
                          detail::PreGenExecutor{properties_->device_ref(),
                                                 cerenkov_->device_ref(),
                                                 scintillation_->device_ref(),
                                                 gen_state,
                                                 offsets}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);

    // Compact the buffers
    offsets.cerenkov = this->remove_if_invalid(
        gen_state.cerenkov, offsets.cerenkov, state.size(), stream);
    offsets.scintillation = this->remove_if_invalid(
        gen_state.scintillation, offsets.scintillation, state.size(), stream);
}

//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 */
template<StepPoint P>
size_type
PreGenAction<P>::remove_if_invalid(ItemsRef<MemSpace::device> const& buffer,
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
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class PreGenAction<StepPoint::pre>;
template class PreGenAction<StepPoint::post>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

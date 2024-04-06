//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenAction.cc
//---------------------------------------------------------------------------//
#include "PreGenAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
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
 * Construct with action ID, optical properties, and storage.
 */
template<StepPoint P>
PreGenAction<P>::PreGenAction(ActionId id,
                              SPConstProperties properties,
                              SPConstCerenkov cerenkov,
                              SPConstScintillation scintillation,
                              SPGenStorage storage)
    : id_(id)
    , properties_(std::move(properties))
    , cerenkov_(std::move(cerenkov))
    , scintillation_(std::move(scintillation))
    , storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(scintillation_ || (cerenkov_ && properties_));
    CELER_EXPECT(!cerenkov == !properties);
    CELER_EXPECT(storage_);

    offsets_.resize(storage_->obj.num_streams(), {});
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
template<StepPoint P>
std::string PreGenAction<P>::description() const
{
    std::string result = "gather ";
    result += P == StepPoint::pre ? "pre" : "post";
    result += "-step data to generate optical distributions";
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data.
 */
template<>
void PreGenAction<StepPoint::pre>::execute(CoreParams const& params,
                                           CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenGatherExecutor{storage_->obj.state<MemSpace::native>(
            state.stream_id(), state.size())});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<>
void PreGenAction<StepPoint::post>::execute(CoreParams const& params,
                                            CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<>
template<MemSpace M>
void PreGenAction<StepPoint::post>::execute_impl(CoreParams const& core_params,
                                                 CoreState<M>& core_state) const
{
    size_type size = core_state.size();
    StreamId stream = core_state.stream_id();
    auto& offsets = offsets_[stream.get()];
    auto const& state = storage_->obj.state<M>(stream, size);

    CELER_VALIDATE(offsets.cerenkov + size <= state.cerenkov.size(),
                   << "insufficient capacity (" << state.cerenkov.size()
                   << ") for buffered Cerenkov distribution data (total "
                      "capacity requirement of "
                   << offsets.cerenkov + size << ")");
    CELER_VALIDATE(offsets.scintillation + size <= state.scintillation.size(),
                   << "insufficient capacity (" << state.scintillation.size()
                   << ") for buffered scintillation distribution data (total "
                      "capacity requirement of "
                   << offsets.scintillation + size << ")");

    // Generate the optical distribution data
    this->pre_generate(core_params, core_state);

    // Compact the buffers
    offsets.cerenkov = this->remove_if_invalid(
        state.cerenkov, offsets.cerenkov, size, stream);
    offsets.scintillation = this->remove_if_invalid(
        state.scintillation, offsets.scintillation, size, stream);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<>
void PreGenAction<StepPoint::post>::pre_generate(CoreParams const& core_params,
                                                 CoreStateHost& core_state) const
{
    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::PreGenExecutor{properties_->host_ref(),
                               cerenkov_->host_ref(),
                               scintillation_->host_ref(),
                               storage_->obj.state<MemSpace::native>(
                                   core_state.stream_id(), core_state.size()),
                               offsets_[core_state.stream_id().get()]}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 */
template<>
size_type PreGenAction<StepPoint::post>::remove_if_invalid(
    ItemsRef<MemSpace::host> const& buffer,
    size_type offset,
    size_type size,
    StreamId) const
{
    auto* start = static_cast<OpticalDistributionData*>(buffer.data());
    auto* stop
        = std::remove_if(start + offset, start + offset + size, IsInvalid{});
    return stop - start;
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<StepPoint P>
void PreGenAction<P>::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

template<>
size_type PreGenAction<StepPoint::post>::remove_if_invalid(
    ItemsRef<MemSpace::device> const&, size_type, size_type, StreamId) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class PreGenAction<StepPoint::pre>;
template class PreGenAction<StepPoint::post>;
template void
PreGenAction<StepPoint::post>::execute_impl(CoreParams const&,
                                            CoreState<MemSpace::host>&) const;
template void
PreGenAction<StepPoint::post>::execute_impl(CoreParams const&,
                                            CoreState<MemSpace::device>&) const;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

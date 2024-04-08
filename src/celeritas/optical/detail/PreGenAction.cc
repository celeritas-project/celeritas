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

#include "OpticalGenStorage.hh"
#include "PreGenExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, optical properties, and storage.
 */
PreGenAction::PreGenAction(ActionId id,
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
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string PreGenAction::description() const
{
    return "generate Cerenkov and scintillation optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data on host.
 */
void PreGenAction::execute(CoreParams const& params, CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data on device.
 */
void PreGenAction::execute(CoreParams const& params,
                           CoreStateDevice& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void PreGenAction::execute_impl(CoreParams const& core_params,
                                CoreState<M>& core_state) const
{
    size_type state_size = core_state.size();
    StreamId stream = core_state.stream_id();
    auto& buffer_size = storage_->size[stream.get()];
    auto const& state = storage_->obj.state<M>(stream, state_size);

    CELER_VALIDATE(buffer_size.cerenkov + state_size <= state.cerenkov.size(),
                   << "insufficient capacity (" << state.cerenkov.size()
                   << ") for buffered Cerenkov distribution data (total "
                      "capacity requirement of "
                   << buffer_size.cerenkov + state_size << ")");
    CELER_VALIDATE(
        buffer_size.scintillation + state_size <= state.scintillation.size(),
        << "insufficient capacity (" << state.scintillation.size()
        << ") for buffered scintillation distribution data (total "
           "capacity requirement of "
        << buffer_size.scintillation + state_size << ")");

    // Generate the optical distribution data
    this->pre_generate(core_params, core_state);

    // Compact the buffers
    buffer_size.cerenkov = this->remove_if_invalid(
        state.cerenkov, buffer_size.cerenkov, state_size, stream);
    buffer_size.scintillation = this->remove_if_invalid(
        state.scintillation, buffer_size.scintillation, state_size, stream);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
void PreGenAction::pre_generate(CoreParams const& core_params,
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
                               storage_->size[core_state.stream_id().get()]}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 */
size_type
PreGenAction::remove_if_invalid(ItemsRef<MemSpace::host> const& buffer,
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
void PreGenAction::pre_generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

size_type PreGenAction::remove_if_invalid(ItemsRef<MemSpace::device> const&,
                                          size_type,
                                          size_type,
                                          StreamId) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

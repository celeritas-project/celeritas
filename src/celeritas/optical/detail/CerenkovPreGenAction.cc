//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovPreGenAction.cc
//---------------------------------------------------------------------------//
#include "CerenkovPreGenAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
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
 * Construct with action ID, optical properties, and storage.
 */
CerenkovPreGenAction::CerenkovPreGenAction(ActionId id,
                                           SPConstProperties properties,
                                           SPConstCerenkov cerenkov,
                                           SPGenStorage storage)
    : id_(id)
    , properties_(std::move(properties))
    , cerenkov_(std::move(cerenkov))
    , storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(cerenkov_ && properties_);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view CerenkovPreGenAction::description() const
{
    return "generate Cerenkov optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void CerenkovPreGenAction::execute(CoreParams const& params,
                                   CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void CerenkovPreGenAction::execute(CoreParams const& params,
                                   CoreStateDevice& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void CerenkovPreGenAction::execute_impl(CoreParams const& core_params,
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

    // Generate the optical distribution data
    this->pre_generate(core_params, core_state);

    // Compact the buffer
    buffer_size.cerenkov = remove_if_invalid(
        state.cerenkov, buffer_size.cerenkov, state_size, stream);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
void CerenkovPreGenAction::pre_generate(CoreParams const& core_params,
                                        CoreStateHost& core_state) const
{
    TrackExecutor execute{core_params.ptr<MemSpace::native>(),
                          core_state.ptr(),
                          detail::CerenkovPreGenExecutor{
                              properties_->host_ref(),
                              cerenkov_->host_ref(),
                              storage_->obj.state<MemSpace::native>(
                                  core_state.stream_id(), core_state.size()),
                              storage_->size[core_state.stream_id().get()]}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void CerenkovPreGenAction::pre_generate(CoreParams const&,
                                        CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

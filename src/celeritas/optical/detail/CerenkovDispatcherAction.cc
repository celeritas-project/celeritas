//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovDispatcherAction.cc
//---------------------------------------------------------------------------//
#include "CerenkovDispatcherAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/MaterialPropertyParams.hh"

#include "CerenkovDispatcherExecutor.hh"
#include "DispatcherParams.hh"
#include "OpticalGenAlgorithms.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data ID, optical properties.
 */
CerenkovDispatcherAction::CerenkovDispatcherAction(ActionId id,
                                                   AuxId data_id,
                                                   SPConstProperties properties,
                                                   SPConstCerenkov cerenkov)
    : id_(id)
    , data_id_{data_id}
    , properties_(std::move(properties))
    , cerenkov_(std::move(cerenkov))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(data_id_);
    CELER_EXPECT(cerenkov_ && properties_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view CerenkovDispatcherAction::description() const
{
    return "generate Cerenkov optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void CerenkovDispatcherAction::execute(CoreParams const& params,
                                       CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void CerenkovDispatcherAction::execute(CoreParams const& params,
                                       CoreStateDevice& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void CerenkovDispatcherAction::execute_impl(CoreParams const& core_params,
                                            CoreState<M>& core_state) const
{
    auto& state = get<OpticalGenState<M>>(core_state.aux(), data_id_);
    auto& buffer = state.store.ref().cerenkov;
    auto& buffer_size = state.buffer_size.cerenkov;

    CELER_VALIDATE(buffer_size + core_state.size() <= buffer.size(),
                   << "insufficient capacity (" << buffer.size()
                   << ") for buffered Cerenkov distribution data (total "
                      "capacity requirement of "
                   << buffer_size + core_state.size() << ")");

    // Generate the optical distribution data
    this->pre_generate(core_params, core_state);

    // Compact the buffer
    buffer_size = remove_if_invalid(
        buffer, buffer_size, core_state.size(), core_state.stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
void CerenkovDispatcherAction::pre_generate(CoreParams const& core_params,
                                            CoreStateHost& core_state) const
{
    auto& state
        = get<OpticalGenState<MemSpace::native>>(core_state.aux(), data_id_);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::CerenkovDispatcherExecutor{properties_->host_ref(),
                                           cerenkov_->host_ref(),
                                           state.store.ref(),
                                           state.buffer_size}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void CerenkovDispatcherAction::pre_generate(CoreParams const&,
                                            CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

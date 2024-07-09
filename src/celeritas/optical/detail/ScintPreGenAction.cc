//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/ScintPreGenAction.cc
//---------------------------------------------------------------------------//
#include "ScintPreGenAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "OpticalGenAlgorithms.hh"
#include "PreGenParams.hh"
#include "ScintPreGenExecutor.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data ID, optical properties.
 */
ScintPreGenAction::ScintPreGenAction(ActionId id,
                                     AuxId data_id,
                                     SPConstScintillation scintillation)
    : id_(id), data_id_{data_id}, scintillation_(std::move(scintillation))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(data_id_);
    CELER_EXPECT(scintillation_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view ScintPreGenAction::description() const
{
    return "generate scintillation optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ScintPreGenAction::execute(CoreParams const& params,
                                CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ScintPreGenAction::execute(CoreParams const& params,
                                CoreStateDevice& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void ScintPreGenAction::execute_impl(CoreParams const& core_params,
                                     CoreState<M>& core_state) const
{
    auto& state = get<OpticalGenState<M>>(core_state.aux(), data_id_);
    auto& buffer = state.store.ref().scintillation;
    auto& buffer_size = state.buffer_size.scintillation;

    CELER_VALIDATE(buffer_size + core_state.size() <= buffer.size(),
                   << "insufficient capacity (" << buffer.size()
                   << ") for buffered scintillation distribution data (total "
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
void ScintPreGenAction::pre_generate(CoreParams const& core_params,
                                     CoreStateHost& core_state) const
{
    auto& state
        = get<OpticalGenState<MemSpace::native>>(core_state.aux(), data_id_);
    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::ScintPreGenExecutor{
            scintillation_->host_ref(), state.store.ref(), state.buffer_size}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ScintPreGenAction::pre_generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

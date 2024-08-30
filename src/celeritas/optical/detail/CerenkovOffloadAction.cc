//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovOffloadAction.cc
//---------------------------------------------------------------------------//
#include "CerenkovOffloadAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "CerenkovOffloadExecutor.hh"
#include "OffloadParams.hh"
#include "OpticalGenAlgorithms.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data ID, optical material.
 */
CerenkovOffloadAction::CerenkovOffloadAction(ActionId id,
                                             AuxId data_id,
                                             SPConstMaterial material,
                                             SPConstCerenkov cerenkov)
    : id_(id)
    , data_id_{data_id}
    , material_(std::move(material))
    , cerenkov_(std::move(cerenkov))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(data_id_);
    CELER_EXPECT(cerenkov_ && material_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view CerenkovOffloadAction::description() const
{
    return "generate Cerenkov optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void CerenkovOffloadAction::execute(CoreParams const& params,
                                    CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void CerenkovOffloadAction::execute(CoreParams const& params,
                                    CoreStateDevice& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void CerenkovOffloadAction::execute_impl(CoreParams const& core_params,
                                         CoreState<M>& core_state) const
{
    auto& state = get<OpticalOffloadState<M>>(core_state.aux(), data_id_);
    auto& buffer = state.store.ref().cerenkov;
    auto& buffer_size = state.buffer_size.cerenkov;

    CELER_VALIDATE(buffer_size + core_state.size() <= buffer.size(),
                   << "insufficient capacity (" << buffer.size()
                   << ") for buffered Cerenkov distribution data (total "
                      "capacity requirement of "
                   << buffer_size + core_state.size() << ")");

    // Generate the optical distribution data
    this->pre_generate(core_params, core_state);

    // Compact the buffer, returning the total number of valid distributions
    size_type start = buffer_size;
    buffer_size = remove_if_invalid(
        buffer, start, start + core_state.size(), core_state.stream_id());

    // Count the number of optical photons that would be generated from the
    // distributions created in this step
    state.buffer_size.num_primaries += count_num_photons(
        buffer, start, buffer_size, core_state.stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
void CerenkovOffloadAction::pre_generate(CoreParams const& core_params,
                                         CoreStateHost& core_state) const
{
    auto& state = get<OpticalOffloadState<MemSpace::native>>(core_state.aux(),
                                                             data_id_);

    TrackExecutor execute{core_params.ptr<MemSpace::native>(),
                          core_state.ptr(),
                          detail::CerenkovOffloadExecutor{material_->host_ref(),
                                                          cerenkov_->host_ref(),
                                                          state.store.ref(),
                                                          state.buffer_size}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void CerenkovOffloadAction::pre_generate(CoreParams const&,
                                         CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

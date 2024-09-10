//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "CerenkovGeneratorAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/TrackData.hh"

#include "CerenkovGeneratorExecutor.hh"
#include "OffloadParams.hh"
#include "OpticalGenAlgorithms.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data IDs, and optical properties.
 */
CerenkovGeneratorAction::CerenkovGeneratorAction(ActionId id,
                                                 AuxId offload_id,
                                                 AuxId optical_id,
                                                 SPConstMaterial material,
                                                 SPConstCerenkov cerenkov,
                                                 size_type auto_flush)
    : id_(id)
    , offload_id_{offload_id}
    , optical_id_{optical_id}
    , material_(std::move(material))
    , cerenkov_(std::move(cerenkov))
    , auto_flush_(auto_flush)
{
    CELER_EXPECT(id_);
    CELER_EXPECT(offload_id_);
    CELER_EXPECT(optical_id_);
    CELER_EXPECT(cerenkov_);
    CELER_EXPECT(material_);
    CELER_EXPECT(auto_flush_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view CerenkovGeneratorAction::description() const
{
    return "generate Cerenkov photons from optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void CerenkovGeneratorAction::step(CoreParams const& params,
                                   CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void CerenkovGeneratorAction::step(CoreParams const& params,
                                   CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photon primaries from Cerenkov distribution data.
 */
template<MemSpace M>
void CerenkovGeneratorAction::step_impl(CoreParams const& core_params,
                                        CoreState<M>& core_state) const
{
    auto& offload_state
        = get<OpticalOffloadState<M>>(core_state.aux(), offload_id_);
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), optical_id_);

    auto& num_primaries = optical_state.counters().num_primaries;
    auto& num_new_primaries = offload_state.buffer_size.num_primaries;

    if (num_primaries + num_new_primaries < auto_flush_)
        return;

    auto primaries_size = optical_state.ref().init.primaries.size();
    CELER_VALIDATE(num_primaries + num_new_primaries <= primaries_size,
                   << "insufficient capacity (" << primaries_size
                   << ") for optical photon primaries (total capacity "
                      "requirement of "
                   << num_primaries + num_new_primaries << ")");

    auto& offload = offload_state.store.ref();
    auto& buffer_size = offload_state.buffer_size.cerenkov;
    CELER_ASSERT(buffer_size > 0);

    // Calculate the cumulative sum of the number of photons in the buffered
    // distributions. These values are used to determine which thread will
    // generate primaries from which distribution
    auto count = inclusive_scan_primaries(
        offload.cerenkov, offload.offsets, buffer_size, core_state.stream_id());

    // Generate the optical primaries from the distribution data
    this->generate(core_params, core_state);

    num_primaries += count;
    num_new_primaries -= count;
    buffer_size = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photon primaries.
 */
void CerenkovGeneratorAction::generate(CoreParams const& core_params,
                                       CoreStateHost& core_state) const
{
    auto& offload_state = get<OpticalOffloadState<MemSpace::native>>(
        core_state.aux(), offload_id_);
    auto& optical_state = get<optical::CoreState<MemSpace::native>>(
        core_state.aux(), optical_id_);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::CerenkovGeneratorExecutor{core_state.ptr(),
                                          material_->host_ref(),
                                          cerenkov_->host_ref(),
                                          offload_state.store.ref(),
                                          optical_state.ptr(),
                                          offload_state.buffer_size}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void CerenkovGeneratorAction::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

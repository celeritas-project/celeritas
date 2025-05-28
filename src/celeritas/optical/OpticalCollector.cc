//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.cc
//---------------------------------------------------------------------------//
#include "OpticalCollector.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "CoreParams.hh"
#include "MaterialParams.hh"
#include "gen/CherenkovParams.hh"
#include "gen/OffloadData.hh"
#include "gen/ScintillationParams.hh"
#include "gen/detail/CherenkovOffloadAction.hh"
#include "gen/detail/OffloadGatherAction.hh"
#include "gen/detail/ScintOffloadAction.hh"

#include "detail/OpticalLaunchAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with core data and optical data.
 *
 * This adds several actions and auxiliary data to the registry.
 */
OpticalCollector::OpticalCollector(CoreParams const& core, Input&& inp)
{
    CELER_EXPECT(inp);

    // Action to gather pre-step data needed to generate optical distributions
    gather_ = detail::OffloadGatherAction::make_and_insert(core);

    // The offload, generator, and launch actions much be created in a specific
    // order but require auxiliary data IDs from actions created later.
    // Precalculate the IDs for teh generator and optical statue aux data.
    size_type num_gen = !!inp.cherenkov + !!inp.scintillation;
    auto gen_aux_id = core.aux_reg()->next_id();
    auto optical_aux_id = core.aux_reg()->next_id() + num_gen;

    ActionRegistry& actions = *core.action_reg();

    if (inp.cherenkov)
    {
        // Action to generate Cherenkov optical distributions
        cherenkov_offload_ = std::make_shared<detail::CherenkovOffloadAction>(
            actions.next_id(),
            gather_->aux_id(),
            gen_aux_id++,
            optical_aux_id,
            inp.material,
            inp.cherenkov);
        actions.insert(cherenkov_offload_);
    }
    if (inp.scintillation)
    {
        // Action to generate scintillation optical distributions
        scint_offload_
            = std::make_shared<detail::ScintOffloadAction>(actions.next_id(),
                                                           gather_->aux_id(),
                                                           gen_aux_id++,
                                                           optical_aux_id,
                                                           inp.scintillation);
        actions.insert(scint_offload_);
    }

    if (inp.cherenkov)
    {
        // Action to generate Cherenkov primaries
        GeneratorAction<GT::cherenkov>::Input gen_inp;
        gen_inp.optical_id = optical_aux_id;
        gen_inp.material = inp.material;
        gen_inp.shared = inp.cherenkov;
        gen_inp.auto_flush = inp.auto_flush;
        gen_inp.buffer_capacity = inp.buffer_capacity;
        cherenkov_generate_ = GeneratorAction<GT::cherenkov>::make_and_insert(
            core, std::move(gen_inp));
    }
    if (inp.scintillation)
    {
        // Action to generate scintillation primaries
        GeneratorAction<GT::scintillation>::Input gen_inp;
        gen_inp.optical_id = optical_aux_id;
        gen_inp.material = inp.material;
        gen_inp.shared = inp.scintillation;
        gen_inp.auto_flush = inp.auto_flush;
        gen_inp.buffer_capacity = inp.buffer_capacity;
        scint_generate_ = GeneratorAction<GT::scintillation>::make_and_insert(
            core, std::move(gen_inp));
    }

    // Create launch action with optical params+state and access to gen data
    detail::OpticalLaunchAction::Input la_inp;
    la_inp.model_builders = std::move(inp.model_builders);
    la_inp.material = inp.material;
    la_inp.num_track_slots = inp.num_track_slots;
    la_inp.initializer_capacity = inp.initializer_capacity;
    launch_ = detail::OpticalLaunchAction::make_and_insert(core,
                                                           std::move(la_inp));

    // Launch action must be *after* offload and generator actions
    CELER_ENSURE(!cherenkov_offload_
                 || launch_->action_id() > cherenkov_offload_->action_id());
    CELER_ENSURE(!scint_offload_
                 || launch_->action_id() > scint_offload_->action_id());
    CELER_ENSURE(!cherenkov_generate_
                 || launch_->action_id() > cherenkov_generate_->action_id());
    CELER_ENSURE(!scint_generate_
                 || launch_->action_id() > scint_generate_->action_id());
    CELER_ENSURE(this->optical_aux_id() == optical_aux_id);
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical core state data.
 */
AuxId OpticalCollector::optical_aux_id() const
{
    return launch_->aux_id();
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical Cherenkov offload data.
 */
AuxId OpticalCollector::cherenkov_aux_id() const
{
    return cherenkov_generate_ ? cherenkov_generate_->aux_id() : AuxId{};
}

//---------------------------------------------------------------------------//
/*!
 * Aux ID for optical scintillation offload data.
 */
AuxId OpticalCollector::scintillation_aux_id() const
{
    return scint_generate_ ? scint_generate_->aux_id() : AuxId{};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

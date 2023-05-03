//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/RootOutput.cc
//---------------------------------------------------------------------------//
#include "RootOutput.hh"

#include <string>
#include <vector>
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"

#include "RunnerInput.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Store input information to the ROOT MC truth output file.
 */
void write_to_root(RunnerInput const& cargs,
                   celeritas::RootFileManager* root_manager)
{
    CELER_EXPECT(cargs);

    auto& args = const_cast<RunnerInput&>(cargs);
    auto tree_input = root_manager->make_tree("input", "input");

    // Problem definition
    tree_input->Branch("geometry_filename", &args.geometry_filename);
    tree_input->Branch("physics_filename", &args.physics_filename);
    tree_input->Branch("hepmc3_filename", &args.hepmc3_filename);

    // Control
    tree_input->Branch("seed", &args.seed);
    tree_input->Branch("max_num_tracks", &args.max_num_tracks);
    tree_input->Branch("max_steps", &args.max_steps);
    tree_input->Branch("initializer_capacity", &args.initializer_capacity);
    tree_input->Branch("max_events", &args.max_events);
    tree_input->Branch("secondary_stack_factor", &args.secondary_stack_factor);
    tree_input->Branch("use_device", &args.use_device);
    tree_input->Branch("sync", &args.sync);
    tree_input->Branch("step_limiter", &args.step_limiter);
    tree_input->Branch("combined_brem", &args.brem_combined);

    // Physics list
    auto& phys = args.geant_options;
    tree_input->Branch("coulomb_scattering", &phys.coulomb_scattering);
    tree_input->Branch("compton_scattering", &phys.compton_scattering);
    tree_input->Branch("photoelectric", &phys.photoelectric);
    tree_input->Branch("rayleigh_scattering", &phys.rayleigh_scattering);
    tree_input->Branch("gamma_conversion", &phys.gamma_conversion);
    tree_input->Branch("gamma_general", &phys.gamma_general);
    tree_input->Branch("ionization", &phys.ionization);
    tree_input->Branch("annihilation", &phys.annihilation);
    tree_input->Branch("brems", &phys.brems);
    tree_input->Branch(
        "brems_selection", &phys.brems_selection, "brems_selection/I");
    tree_input->Branch("msc", &phys.msc, "msc/I");
    tree_input->Branch("relaxation", &phys.relaxation, "relaxation/I");
    tree_input->Branch("eloss_fluctuation", &phys.eloss_fluctuation);
    tree_input->Branch("lpm", &phys.lpm);
    tree_input->Branch("integral_approach", &phys.integral_approach);
    tree_input->Branch("min_energy", &phys.min_energy.value());
    tree_input->Branch("max_energy", &phys.max_energy.value());
    tree_input->Branch("linear_loss_limit", &phys.linear_loss_limit);
    tree_input->Branch("lowest_electron_energy",
                       &phys.lowest_electron_energy.value());
    tree_input->Branch("msc_range_factor", &phys.msc_range_factor);
    tree_input->Branch("msc_safety_factor", &phys.msc_safety_factor);
    tree_input->Branch("msc_lambda_limit", &phys.msc_lambda_limit);
    tree_input->Branch("apply_cuts", &phys.apply_cuts);

    // TODO Add magnetic field information?

    // Fill tree (writing happens at destruction)
    tree_input->Fill();
}

//---------------------------------------------------------------------------//
/*!
 * Store CoreParams data to the ROOT MC truth output file.
 *
 * \note
 * Currently only storing the action labels so their IDs can be identified. If
 * other parameters are needed for future debugging/analyses, this function can
 * easily be expanded.
 */
void write_to_root(celeritas::CoreParams const& core_params,
                   celeritas::RootFileManager* root_manager)
{
    auto const& action_reg = *core_params.action_reg();

    // Initialize CoreParams TTree
    auto tree_params = root_manager->make_tree("core_params", "core_params");

    // Store labels
    std::vector<std::string> action_labels;
    action_labels.resize(action_reg.num_actions());
    for (auto const id : celeritas::range(action_reg.num_actions()))
    {
        action_labels[id] = action_reg.id_to_label(celeritas::ActionId{id});
    }

    // Set up action labels branch, fill the TTree and write it
    /*
     * The decision to store a vector instead of making a tree entry for
     * each label is to simplify the reading of the information. Calling
     * action_labels->at(action_id) after loading the first (and only) tree
     * entry is much simpler than:
     * tree->GetEntry(action_id);
     * tree->GetLeaf("action_label")->GetValue();
     */
    tree_params->Branch("action_labels", &action_labels);
    tree_params->Fill();  // Writing happens at destruction
}

//---------------------------------------------------------------------------//
}  // namespace demo_loop

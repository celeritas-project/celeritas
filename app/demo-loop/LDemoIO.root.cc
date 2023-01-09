//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.root.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <string>
#include <vector>
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

#include "corecel/io/Logger.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Store input information to the ROOT MC truth output file.
 */
void to_root(std::shared_ptr<celeritas::RootFileManager>& root_manager,
             LDemoArgs&                                   args)
{
    CELER_EXPECT(root_manager);
    CELER_EXPECT(args);

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
    tree_input->Branch("enable_diagnostics", &args.enable_diagnostics);
    tree_input->Branch("use_device", &args.use_device);
    tree_input->Branch("sync", &args.sync);
    tree_input->Branch("step_limiter", &args.step_limiter);

    // Options for physics processes and models
    tree_input->Branch("combined_brem", &args.brem_combined);

    // TODO Add magnetic field information?

    // Fill tree and write it to file
    tree_input->Fill();
    tree_input->Write();
}

//---------------------------------------------------------------------------//
/*!
 * Store CoreParams data to the ROOT MC truth output file. Currently it only
 * stores the action labels so they can be identified during the data analysis.
 */
void store_core_params(std::shared_ptr<celeritas::RootFileManager>& root_manager,
                       celeritas::CoreParams core_params)
{
    CELER_EXPECT(root_manager);

    auto action_reg = core_params.action_reg();
    CELER_EXPECT(action_reg);

    auto tree_params = root_manager->make_tree("core_params", "core_params");

    std::vector<std::string> action_labels;
    tree_params->Branch("action_labels", &action_labels);

    action_labels.resize(action_reg->num_actions());

    for (const auto id : celeritas::range(action_reg->num_actions()))
    {
        action_labels[id] = action_reg->id_to_label(celeritas::ActionId{id});
    }

    tree_params->Fill();
    tree_params->Write();
}

//---------------------------------------------------------------------------//
} // namespace demo_loop

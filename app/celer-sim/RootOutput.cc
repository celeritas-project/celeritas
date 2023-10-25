//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RootOutput.cc
//---------------------------------------------------------------------------//
#include "RootOutput.hh"

#include <string>
#include <vector>
#include <TTree.h>

#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"

#include "RunnerInput.hh"

#if CELERITAS_USE_JSON
#    include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"

#    include "RunnerInputIO.json.hh"
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Store input information to the ROOT MC truth output file.
 */
void write_to_root(RunnerInput const& cargs, RootFileManager* root_manager)
{
    CELER_EXPECT(cargs);
    CELER_EXPECT(root_manager);

#if CELERITAS_USE_JSON
    std::string str_input(nlohmann::json(cargs).dump());
    std::string str_phys(nlohmann::json(cargs.physics_options).dump());

    auto tree_input = root_manager->make_tree("input", "input");
    tree_input->Branch("input", &str_input);
    tree_input->Branch("physics_options", &str_phys);
    tree_input->Fill();  // Writing happens at destruction
#else
    CELER_NOT_CONFIGURED("nlohmann_json");
#endif
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
void write_to_root(CoreParams const& core_params, RootFileManager* root_manager)
{
    auto const& action_reg = *core_params.action_reg();

    // Initialize CoreParams TTree
    auto tree_params = root_manager->make_tree("core_params", "core_params");

    // Store labels
    std::vector<std::string> action_labels;
    action_labels.resize(action_reg.num_actions());
    for (auto const id : range(action_reg.num_actions()))
    {
        action_labels[id] = action_reg.id_to_label(ActionId{id});
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
}  // namespace app
}  // namespace celeritas

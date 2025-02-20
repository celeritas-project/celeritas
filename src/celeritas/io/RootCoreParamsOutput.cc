//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootCoreParamsOutput.cc
//---------------------------------------------------------------------------//
#include "RootCoreParamsOutput.hh"

#include <string>
#include <vector>
#include <TTree.h>

#include "corecel/cont/Range.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/ext/RootFileManager.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store CoreParams data to the ROOT MC truth output file.
 *
 * \note
 * Currently only storing the action labels so their IDs can be identified. If
 * other parameters are needed for future debugging/analyses, this function can
 * easily be expanded.
 */
void write_to_root(ActionRegistry const& action_reg,
                   RootFileManager* root_manager)
{
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
}  // namespace celeritas

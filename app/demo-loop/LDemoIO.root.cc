//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.root.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

namespace demo_loop
{
//---------------------------------------------------------------------------//
void to_root(std::shared_ptr<celeritas::RootFileManager> root_manager,
             LDemoArgs&                                  args)
{
    CELER_ASSERT(args);
    CELER_ASSERT(root_manager);

    // So far RootFileManager has only a single TFile instance and ROOT knows
    // that this new TTree has to be attached to this single TFile. If we
    // expand it to a RootFileManager::tfile_[tid] we'd need to specify the
    // correct TFile* by invoking TTree("name", "title", TFile*).
    std::unique_ptr<TTree> tree_input
        = std::make_unique<TTree>("input", "input");

    // Problem definition
    tree_input->Branch("geometry_filename", &args.geometry_filename);
    tree_input->Branch("physics_filename", &args.physics_filename);
    tree_input->Branch("hepmc3_filename", &args.hepmc3_filename);

    // Control
    tree_input->Branch("seed", &args.seed);
    // {
    // Celeritas' type aliases have to be manually casted to be stored properly
    // See leaflist types: https://root.cern.ch/doc/master/classTBranch.html
    tree_input->Branch(
        "max_num_tracks", &args.max_num_tracks, "max_num_tracks/l");
    tree_input->Branch("max_steps", &args.max_steps, "max_num_steps/l");

    tree_input->Branch("initializer_capacity",
                       &args.initializer_capacity,
                       "initializer_capacity/l");
    tree_input->Branch("max_events", &args.max_events, "max_events/l");
    tree_input->Branch("secondary_stack_factor",
                       &args.secondary_stack_factor,
                       "secondary_stack_factor/D");
    // }
    tree_input->Branch("enable_diagnostics", &args.enable_diagnostics);
    tree_input->Branch("use_device", &args.use_device);
    tree_input->Branch("sync", &args.sync);
    tree_input->Branch("step_limiter", &args.step_limiter);

    // Options for physics processes and models
    tree_input->Branch("combined_brem", &args.brem_combined);

    // TODO Add magnetic field information?

    tree_input->Fill();
    tree_input->Write();
}
//---------------------------------------------------------------------------//
} // namespace demo_loop

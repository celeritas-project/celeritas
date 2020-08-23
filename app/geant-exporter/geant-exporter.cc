//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geant-exporter.cc
//! \brief Geant4 particle definition and physics tables exporter app
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <G4RunManager.hh>
#include <G4UImanager.hh>
#include <FTFP_BERT.hh>
#include <G4VModularPhysicsList.hh>
#include <G4GenericPhysicsList.hh>
#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"
#include "io/GeantParticle.hh"

using celeritas::GeantParticle;
using namespace geant_exporter;
using std::cout;
using std::endl;

//---------------------------------------------------------------------------//
/*!
 * Write particle table data to ROOT.
 *
 * The ROOT file must be open before this call.
 */
void store_particles(TFile* root_file, G4ParticleTable* particle_table)
{
    REQUIRE(root_file);
    REQUIRE(particle_table);

    cout << "Exporting particles..." << endl;
    TTree tree_particles("particles", "particles");

    // Create temporary particle
    GeantParticle particle;
    tree_particles.Branch("GeantParticle", &particle);

    G4ParticleTable::G4PTblDicIterator* particle_iterator
        = G4ParticleTable::GetParticleTable()->GetIterator();
    particle_iterator->reset();

    while ((*particle_iterator)())
    {
        G4ParticleDefinition* g4_particle_def = particle_iterator->value();

        // Skip the Geantino: shares "dummy" pdg encoding (0) with GenericIon
        if (g4_particle_def->GetParticleName() == "geantino")
            continue;

        particle.name      = g4_particle_def->GetParticleName();
        particle.pdg       = g4_particle_def->GetPDGEncoding();
        particle.mass      = g4_particle_def->GetPDGMass();
        particle.charge    = g4_particle_def->GetPDGCharge();
        particle.spin      = g4_particle_def->GetPDGSpin();
        particle.lifetime  = g4_particle_def->GetPDGLifeTime();
        particle.is_stable = g4_particle_def->GetPDGStable();

        if (!particle.is_stable)
        {
            // Convert lifetime of unstable particles to seconds
            particle.lifetime /= s;
        }

        tree_particles.Fill();

        cout << "  Added " << g4_particle_def->GetParticleName() << endl;
    }

    root_file->Write();
}

//---------------------------------------------------------------------------//
/*!
 * This application exports particle and physics table data constructed by the
 * selected physics list. Output data is stored into a ROOT file.
 */
int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        // Incorrect number of arguments: print help and exit
        cout << "Usage: " << argv[0] << " geometry.gdml output.root" << endl;
        return EXIT_FAILURE;
    }
    std::string gdml_input_filename  = argv[1];
    std::string root_output_filename = argv[2];

    // >>> Initialize Geant4

    G4RunManager run_manager;

    // Initialize the geometry
    auto detector = std::make_unique<DetectorConstruction>(gdml_input_filename);
    run_manager.SetUserInitialization(detector.release());

    // Load the physics list
    auto physics_constructor = std::make_unique<std::vector<G4String>>();
    physics_constructor->push_back("G4EmStandardPhysics");

    auto physics_list = std::make_unique<G4GenericPhysicsList>(
        physics_constructor.release());

    // For the full Physics List:
    // auto physics_list = std::make_unique<FTFP_BERT>();

    run_manager.SetUserInitialization(physics_list.release());

    // Run a single partlce to generate the physics tables
    auto action_initialization = std::make_unique<ActionInitialization>();
    run_manager.SetUserInitialization(action_initialization.release());

    G4UImanager::GetUIpointer()->ApplyCommand("/run/initialize");
    run_manager.BeamOn(1);

    // >>> Export data

    // Create the ROOT output file
    cout << "Creating " << root_output_filename << "..." << endl;
    TFile root_output(root_output_filename.c_str(), "recreate");

    // Store particle information
    store_particles(&root_output, G4ParticleTable::GetParticleTable());

    cout << "Writing..." << std::flush;
    root_output.Close();

    cout << " done!" << endl;
    return EXIT_SUCCESS;
}

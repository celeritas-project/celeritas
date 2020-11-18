//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geant-exporter.cc
//! Geant4 particle, XS tables, material, and volume data exporter app.
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
#include <G4Material.hh>
#include <G4MaterialTable.hh>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "GeantPhysicsTableWriter.hh"
#include "io/ImportParticle.hh"
#include "io/ImportPhysicsTable.hh"
#include "io/GdmlGeometryMap.hh"
#include "base/Range.hh"

using namespace geant_exporter;
using celeritas::elem_id;
using celeritas::GdmlGeometryMap;
using celeritas::ImportElement;
using celeritas::ImportMaterial;
using celeritas::ImportMaterialState;
using celeritas::ImportParticle;
using celeritas::ImportVolume;
using celeritas::mat_id;
using celeritas::real_type;
using celeritas::vol_id;
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
    ImportParticle particle;
    tree_particles.Branch("ImportParticle", &particle);

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    while (particle_iterator())
    {
        G4ParticleDefinition* g4_particle_def = particle_iterator.value();

        // Skip "dummy" particles: generic ion and geantino
        if (g4_particle_def->GetPDGEncoding() == 0)
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
    cout << endl;

    root_file->Write();
}

//---------------------------------------------------------------------------//
/*!
 * Write physics table data to ROOT.
 *
 * The ROOT file must be open before this call.
 */
void store_physics_tables(TFile* root_file, G4ParticleTable* particle_table)
{
    REQUIRE(root_file);
    REQUIRE(particle_table);

    // Start table writer
    GeantPhysicsTableWriter table_writer(root_file);

    cout << "Exporting physics tables..." << endl;

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    while (particle_iterator())
    {
        const G4ParticleDefinition& g4_particle_def
            = *(particle_iterator.value());

        // Skip "dummy" particles: generic ion and geantino
        if (g4_particle_def.GetPDGEncoding() == 0)
            continue;

        std::cout << "=============" << std::endl;
        std::cout << g4_particle_def.GetParticleName() << std::endl;
        std::cout << "=============" << std::endl;

        const G4ProcessVector& process_list
            = *(g4_particle_def.GetProcessManager()->GetProcessList());

        for (auto j : celeritas::range(process_list.size()))
        {
            const G4VProcess& process = *(process_list)[j];
            table_writer.add_physics_tables(process, g4_particle_def);
        }
    }
    cout << endl;

    root_file->Write();
}

//---------------------------------------------------------------------------//
/*!
 * Recursive loop over all logical volumes.
 *
 * Function called by store_geometry(...)
 */
void loop_volumes(GdmlGeometryMap&       geometry,
                  const G4LogicalVolume& logical_volume)
{
    ImportVolume volume;
    vol_id       volume_id;
    mat_id       material_id;

    volume.name = logical_volume.GetName();
    volume_id   = logical_volume.GetInstanceID();
    material_id = logical_volume.GetMaterialCutsCouple()->GetIndex();

    // Add volume to the global volume map
    geometry.add_volume(volume_id, volume);

    // Map volume to its material
    geometry.link_volume_material(volume_id, material_id);

    // Recursive: repeat for every daughter volume, if there are any
    for (auto i : celeritas::range(logical_volume.GetNoDaughters()))
    {
        loop_volumes(geometry,
                     *logical_volume.GetDaughter(i)->GetLogicalVolume());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4State [G4Material.hh] to ImportMaterialState.
 */
ImportMaterialState to_material_state(const G4State& g4_material_state)
{
    switch (g4_material_state)
    {
        case G4State::kStateUndefined:
            return ImportMaterialState::not_defined;
        case G4State::kStateSolid:
            return ImportMaterialState::solid;
        case G4State::kStateLiquid:
            return ImportMaterialState::liquid;
        case G4State::kStateGas:
            return ImportMaterialState::gas;
    }
    CHECK_UNREACHABLE;
}

//---------------------------------------------------------------------------//
/*!
 * Write material table data to ROOT.
 *
 * The ROOT file must be open before this call.
 */
void store_geometry(TFile*                       root_file,
                    const G4ProductionCutsTable& g4production_cuts,
                    const G4VPhysicalVolume&     world_volume)
{
    REQUIRE(root_file);

    cout << "Exporting material and volume information..." << endl;

    TTree tree_materials("geometry", "geometry");

    // Create geometry map and ROOT branch
    GdmlGeometryMap geometry;
    tree_materials.Branch("GdmlGeometryMap", &geometry);

    // Loop over materials and elements
    for (auto i : celeritas::range(g4production_cuts.GetTableSize()))
    {
        // Fetch material and element list
        const auto& g4material_cuts
            = g4production_cuts.GetMaterialCutsCouple(i);
        const auto& g4material = g4material_cuts->GetMaterial();
        const auto& g4elements = g4material->GetElementVector();

        CHECK(g4material_cuts);
        CHECK(g4material);
        CHECK(g4elements);

        // Populate material information
        ImportMaterial material;
        material.name             = g4material->GetName();
        material.state            = to_material_state(g4material->GetState());
        material.temperature      = g4material->GetTemperature(); // [K]
        material.density          = g4material->GetDensity() / (g / cm3);
        material.electron_density = g4material->GetTotNbOfElectPerVolume()
                                    / (1. / cm3);
        material.atomic_density = g4material->GetTotNbOfAtomsPerVolume()
                                  / (1. / cm3);
        material.radiation_length   = g4material->GetRadlen() / cm;
        material.nuclear_int_length = g4material->GetNuclearInterLength() / cm;

        // Populate element information
        for (auto j : celeritas::range(g4elements->size()))
        {
            const auto& g4element = g4elements->at(j);
            CHECK(g4element);

            ImportElement element;
            element.name                  = g4element->GetName();
            element.atomic_number         = g4element->GetZ();
            element.atomic_mass           = g4element->GetAtomicMassAmu();
            element.radiation_length_tsai = g4element->GetfRadTsai()
                                            / (g / cm2);
            element.coulomb_factor = g4element->GetfCoulomb();

            elem_id   elid = g4element->GetIndex();
            real_type frac = g4material->GetFractionVector()[j];

            // Add element to the global element map
            geometry.add_element(elid, element);

            // Add global element to the map of elements of a given material
            material.elements_fractions.insert({elid, frac});
        }
        // Add material to the global material map
        geometry.add_material(g4material_cuts->GetIndex(), material);
    }

    // Recursive loop over all logical volumes, starting from the world_volume
    // Populate volume information and map volumes with materials
    loop_volumes(geometry, *world_volume.GetLogicalVolume());

    tree_materials.Fill();
    root_file->Write();
}

//---------------------------------------------------------------------------//
/*!
 * This application exports particle information, XS physics tables, material,
 * and volume information constructed by the physics list and geometry.
 *
 * The data is stored into a ROOT file.
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

    /// Initialize Geant4 ///

    G4RunManager run_manager;

    /// Initialize the geometry ///

    auto detector = std::make_unique<DetectorConstruction>(gdml_input_filename);
    // Get world_volume for store_geometry() before releasing detector ptr
    auto world_phys_volume = detector->get_world_volume();
    run_manager.SetUserInitialization(detector.release());

    /// Load the physics list ///

    // User-defined physics list (see PhysicsList.hh)
    // auto physics_list = std::make_unique<PhysicsList>();

    // EM Standard Physics
    auto physics_constructor = std::make_unique<std::vector<G4String>>();
    physics_constructor->push_back("G4EmStandardPhysics");
    auto physics_list = std::make_unique<G4GenericPhysicsList>(
        physics_constructor.release());

    // Full Physics
    // auto physics_list = std::make_unique<FTFP_BERT>();

    run_manager.SetUserInitialization(physics_list.release());

    /// Minimal run to generate the physics tables ///
    auto action_initialization = std::make_unique<ActionInitialization>();
    run_manager.SetUserInitialization(action_initialization.release());
    G4UImanager::GetUIpointer()->ApplyCommand("/run/initialize");
    run_manager.BeamOn(1);

    /// Export data ///

    // Create the ROOT output file
    cout << "Creating " << root_output_filename << "..." << endl;
    TFile root_output(root_output_filename.c_str(), "recreate");

    // Store particle information
    store_particles(&root_output, G4ParticleTable::GetParticleTable());

    // Store physics tables
    store_physics_tables(&root_output, G4ParticleTable::GetParticleTable());

    // Store material and volume information
    store_geometry(&root_output,
                   *G4ProductionCutsTable::GetProductionCutsTable(),
                   *world_phys_volume);

    root_output.Close();

    cout << root_output_filename << " done!" << endl;

    return EXIT_SUCCESS;
}

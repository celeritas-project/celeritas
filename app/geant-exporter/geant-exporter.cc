//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geant-exporter.cc
//! Geant4 pre-processed data exporter app.
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "CeleritasG4Version.hh"
#if CELERITAS_G4_V10
#    include <G4RunManager.hh>
#else
#    include <G4RunManagerFactory.hh>
#endif

#include <G4UImanager.hh>
#include <FTFP_BERT.hh>
#include <G4VModularPhysicsList.hh>
#include <G4GenericPhysicsList.hh>
#include <G4ParticleTable.hh>
#include <G4Material.hh>
#include <G4MaterialTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4Transportation.hh>
#include <G4ProductionCutsTable.hh>
#include <G4RToEConvForElectron.hh>
#include <G4RToEConvForGamma.hh>
#include <G4RToEConvForPositron.hh>
#include <G4RToEConvForProton.hh>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include "base/Range.hh"
#include "comm/Communicator.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"
#include "io/ImportParticle.hh"
#include "io/ImportPhysicsTable.hh"
#include "io/ImportData.hh"
#include "physics/base/PDGNumber.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ImportProcessConverter.hh"
#include "GeantLoggerAdapter.hh"
#include "GeantExceptionHandler.hh"

using namespace geant_exporter;
namespace celer_pdg = celeritas::pdg;
using celeritas::ImportData;
using celeritas::ImportElement;
using celeritas::ImportMatElemComponent;
using celeritas::ImportMaterial;
using celeritas::ImportMaterialState;
using celeritas::ImportParticle;
using celeritas::ImportProductionCut;
using celeritas::ImportVolume;
using std::cout;
using std::endl;

//---------------------------------------------------------------------------//
// Helper functions
//---------------------------------------------------------------------------//

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
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4ProductionCutsIndex [G4ProductionCuts.hh] to the
 * particle's pdg encoding.
 */
int to_pdg(const G4ProductionCutsIndex& index)
{
    switch (index)
    {
        case idxG4GammaCut:
            return celer_pdg::gamma().get();
        case idxG4ElectronCut:
            return celer_pdg::electron().get();
        case idxG4PositronCut:
            return celer_pdg::positron().get();
        case idxG4ProtonCut:
            return celer_pdg::proton().get();
        case NumberOfG4CutIndex:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Store all logical volumes by recursively looping over them.
 *
 * Using a map ensures that volumes are both ordered by volume id and not
 * duplicated.
 * Function called by \c store_volumes(...) .
 */
void loop_volumes(std::map<unsigned int, ImportVolume>& volids_volumes,
                  const G4LogicalVolume&                logical_volume)
{
    // Add volume to the map
    ImportVolume volume;
    volume.material_id = logical_volume.GetMaterialCutsCouple()->GetIndex();
    volume.name        = logical_volume.GetName();
    volume.solid_name  = logical_volume.GetSolid()->GetName();

    volids_volumes.insert({logical_volume.GetInstanceID(), volume});

    // Recursive: repeat for every daughter volume, if there are any
    for (const auto i : celeritas::range(logical_volume.GetNoDaughters()))
    {
        loop_volumes(volids_volumes,
                     *logical_volume.GetDaughter(i)->GetLogicalVolume());
    }
}

//---------------------------------------------------------------------------//
// Store functions
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportParticle vector.
 */
std::vector<ImportParticle> store_particles()
{
    CELER_LOG(status) << "Exporting particles";

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    std::vector<ImportParticle> particles;

    while (particle_iterator())
    {
        const G4ParticleDefinition& g4_particle_def
            = *(particle_iterator.value());

        celeritas::PDGNumber pdg{g4_particle_def.GetPDGEncoding()};
        if (!pdg)
        {
            // Skip "dummy" particles: generic ion and geantino
            continue;
        }

        ImportParticle particle;
        particle.name      = g4_particle_def.GetParticleName();
        particle.pdg       = g4_particle_def.GetPDGEncoding();
        particle.mass      = g4_particle_def.GetPDGMass();
        particle.charge    = g4_particle_def.GetPDGCharge();
        particle.spin      = g4_particle_def.GetPDGSpin();
        particle.lifetime  = g4_particle_def.GetPDGLifeTime();
        particle.is_stable = g4_particle_def.GetPDGStable();

        if (!particle.is_stable)
        {
            // Convert lifetime of unstable particles to seconds
            particle.lifetime /= s;
        }

        particles.push_back(particle);
        CELER_LOG(debug) << "Added " << g4_particle_def.GetParticleName();
    }
    CELER_LOG(info) << "Added " << particles.size() << " particles";
    return particles;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportElement vector.
 */
std::vector<ImportElement> store_elements()
{
    CELER_LOG(status) << "Exporting elements";

    const auto g4element_table = *G4Element::GetElementTable();

    std::vector<ImportElement> elements;
    elements.resize(g4element_table.size());

    // Loop over element data
    for (const auto& g4element : g4element_table)
    {
        CELER_ASSERT(g4element);

        // Add element to vector
        ImportElement element;
        element.name                  = g4element->GetName();
        element.atomic_number         = g4element->GetZ();
        element.atomic_mass           = g4element->GetAtomicMassAmu();
        element.radiation_length_tsai = g4element->GetfRadTsai() / (g / cm2);
        element.coulomb_factor        = g4element->GetfCoulomb();

        elements[g4element->GetIndex()] = element;
    }
    CELER_LOG(info) << "Added " << elements.size() << " elements";
    return elements;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportMaterial vector.
 */
std::vector<ImportMaterial> store_materials()
{
    CELER_LOG(status) << "Exporting materials";

    const auto& g4production_cuts_table
        = *G4ProductionCutsTable::GetProductionCutsTable();

    std::vector<ImportMaterial> materials;
    materials.resize(g4production_cuts_table.GetTableSize());

    // Loop over material data
    for (int i : celeritas::range(g4production_cuts_table.GetTableSize()))
    {
        // Fetch material, element, and production cuts lists
        const auto& g4material_cuts_couple
            = g4production_cuts_table.GetMaterialCutsCouple(i);
        const auto& g4material  = g4material_cuts_couple->GetMaterial();
        const auto& g4elements  = g4material->GetElementVector();
        const auto& g4prod_cuts = g4material_cuts_couple->GetProductionCuts();

        CELER_ASSERT(g4material_cuts_couple);
        CELER_ASSERT(g4material);
        CELER_ASSERT(g4elements);
        CELER_ASSERT(g4prod_cuts);

        // Populate material information
        ImportMaterial material;
        material.name             = g4material->GetName();
        material.state            = to_material_state(g4material->GetState());
        material.temperature      = g4material->GetTemperature(); // [K]
        material.density          = g4material->GetDensity() / (g / cm3);
        material.electron_density = g4material->GetTotNbOfElectPerVolume()
                                    / (1. / cm3);
        material.number_density = g4material->GetTotNbOfAtomsPerVolume()
                                  / (1. / cm3);
        material.radiation_length   = g4material->GetRadlen() / cm;
        material.nuclear_int_length = g4material->GetNuclearInterLength() / cm;

        // Range to energy converters for populating material.cutoffs
        std::unique_ptr<G4VRangeToEnergyConverter>
            range_to_e_converters[NumberOfG4CutIndex];
        range_to_e_converters[idxG4GammaCut]
            = std::make_unique<G4RToEConvForGamma>();
        range_to_e_converters[idxG4ElectronCut]
            = std::make_unique<G4RToEConvForElectron>();
        range_to_e_converters[idxG4PositronCut]
            = std::make_unique<G4RToEConvForPositron>();
        range_to_e_converters[idxG4ProtonCut]
            = std::make_unique<G4RToEConvForProton>();

        // Populate material production cut values
        for (int i : celeritas::range(NumberOfG4CutIndex))
        {
            const auto   g4i   = static_cast<G4ProductionCutsIndex>(i);
            const double range = g4prod_cuts->GetProductionCut(g4i);
            const double energy
                = range_to_e_converters[g4i]->Convert(range, g4material);

            ImportProductionCut cutoffs;
            cutoffs.energy = energy / MeV;
            cutoffs.range  = range / cm;

            material.pdg_cutoffs.insert({to_pdg(g4i), cutoffs});
        }

        // Populate element information for this material
        for (int j : celeritas::range(g4elements->size()))
        {
            const auto& g4element = g4elements->at(j);
            CELER_ASSERT(g4element);

            ImportMatElemComponent elem_comp;
            elem_comp.element_id    = g4element->GetIndex();
            elem_comp.mass_fraction = g4material->GetFractionVector()[j];
            double elem_num_density = g4material->GetVecNbOfAtomsPerVolume()[j]
                                      / (1. / cm3);
            elem_comp.number_fraction = elem_num_density
                                        / material.number_density;

            // Add material's element information
            material.elements.push_back(elem_comp);
        }
        // Add material to vector
        const unsigned int material_id = g4material_cuts_couple->GetIndex();
        CELER_ASSERT(material_id < materials.size());
        materials[g4material_cuts_couple->GetIndex()] = material;
    }

    CELER_LOG(info) << "Added " << materials.size() << " materials";
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportProcess vector.
 */
std::vector<ImportProcess> store_processes()
{
    CELER_LOG(status) << "Exporting physics processes";

    std::vector<ImportProcess> processes;
    ImportProcessConverter     process_writer(TableSelection::minimal);

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    while (particle_iterator())
    {
        const G4ParticleDefinition& g4_particle_def
            = *(particle_iterator.value());

        celeritas::PDGNumber pdg(g4_particle_def.GetPDGEncoding());
        if (!pdg)
        {
            // Skip "dummy" particles: generic ion and geantino
            continue;
        }

        // XXX To reduce ROOT file data size in repo, only export processes for
        // electron/positron/gamma for now. Extend this later.
        if (!(pdg == celer_pdg::electron() || pdg == celer_pdg::positron()
              || pdg == celer_pdg::gamma()))
        {
            // Not e-, e+, or gamma
            continue;
        }

        const G4ProcessVector& process_list
            = *g4_particle_def.GetProcessManager()->GetProcessList();

        for (auto j : celeritas::range(process_list.size()))
        {
            if (dynamic_cast<const G4Transportation*>(process_list[j]))
            {
                // Skip transportation process
                continue;
            }

            if (ImportProcess process
                = process_writer(g4_particle_def, *process_list[j]))
            {
                // Not an empty process, so it was not added in a previous loop
                processes.push_back(std::move(process));
            }
        }
    }
    CELER_LOG(info) << "Added " << processes.size() << " processes";
    return processes;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportVolume vector.
 */
std::vector<ImportVolume> store_volumes(const G4VPhysicalVolume* world_volume)
{
    CELER_LOG(status) << "Exporting volumes";

    std::vector<ImportVolume>            volumes;
    std::map<unsigned int, ImportVolume> volids_volumes;

    // Recursive loop over all logical volumes to populate map<volid, volume>
    loop_volumes(volids_volumes, *world_volume->GetLogicalVolume());

    // Populate vector<ImportVolume>
    for (const auto& key : volids_volumes)
    {
        CELER_ASSERT(key.first == volumes.size());
        volumes.push_back(key.second);
    }

    CELER_LOG(info) << "Added " << volumes.size() << " volumes";
    return volumes;
}

//---------------------------------------------------------------------------//
/*!
 * This application exports particles, processes, models, XS physics
 * tables, material, and volume information constructed by the physics list and
 * loaded by the GDML geometry.
 *
 * The data is stored into a ROOT file as an \c ImportData struct.
 */
int main(int argc, char* argv[])
{
    using namespace celeritas;
    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && Communicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    if (argc != 3)
    {
        // Incorrect number of arguments: print help and exit
        cout << "Usage: " << argv[0] << " geometry.gdml output.root" << endl;
        return 2;
    }
    std::string gdml_input_filename  = argv[1];
    std::string root_output_filename = argv[2];

    //// Initialize Geant4 ////

    CELER_LOG(status) << "Initializing Geant4";

    // Constructing the run manager resets the global log/exception handlers,
    // so it must be done first. The stupid version banner cannot be
    // suppressed.
    std::unique_ptr<G4RunManager> run_manager;

#if CELERITAS_G4_V10
    run_manager = std::make_unique<G4RunManager>();
#else
    run_manager.reset(
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
#endif

    GeantLoggerAdapter    scoped_logger;
    GeantExceptionHandler scoped_exception_handler;

    //// Initialize the geometry ////

    auto detector = std::make_unique<DetectorConstruction>(gdml_input_filename);
    // Get world_volume for store_geometry() before releasing detector ptr
    auto world_phys_volume = detector->get_world_volume();
    run_manager->SetUserInitialization(detector.release());

    //// Load the physics list ////

    if (true)
    {
        // User-defined physics list
        auto physics_list = std::make_unique<PhysicsList>();
        run_manager->SetUserInitialization(physics_list.release());
    }

    // DISABLED
    if (false)
    {
        // EM Standard Physics
        auto physics_constructor = std::make_unique<std::vector<G4String>>();
        physics_constructor->push_back("G4EmStandardPhysics");
        auto physics_list = std::make_unique<G4GenericPhysicsList>(
            physics_constructor.release());
        run_manager->SetUserInitialization(physics_list.release());
    }

    // DISABLED
    if (false)
    {
        // Full Physics
        auto physics_list = std::make_unique<FTFP_BERT>();
        run_manager->SetUserInitialization(physics_list.release());
    }

    //// Minimal run to generate the physics tables ////
    auto action_initialization = std::make_unique<ActionInitialization>();
    run_manager->SetUserInitialization(action_initialization.release());
    G4UImanager::GetUIpointer()->ApplyCommand("/run/initialize");
    run_manager->BeamOn(1);

    //// Export data ////

    CELER_LOG(status) << "Creating ROOT file";
    std::unique_ptr<TFile> root_output(
        TFile::Open(root_output_filename.c_str(), "recreate"));
    CELER_ASSERT(root_output && !root_output->IsZombie());
    CELER_LOG(info) << "Created ROOT output file '" << root_output_filename
                    << "'";

    TTree      tree_data("geant4_data", "geant4_data");
    ImportData import_data;
    TBranch*   branch = tree_data.Branch("ImportData", &import_data);
    CELER_ASSERT(branch);

    // Populate ImportData members
    import_data.particles = store_particles();
    import_data.elements  = store_elements();
    import_data.materials = store_materials();
    import_data.processes = store_processes();
    import_data.volumes   = store_volumes(world_phys_volume);
    CELER_ENSURE(import_data);

    // Write data to disk and close ROOT file
    tree_data.Fill();
    int err_code = root_output->Write();
    CELER_ENSURE(err_code >= 0);
    CELER_LOG(status) << "Closing output file";
    root_output->Close();

    return EXIT_SUCCESS;
}

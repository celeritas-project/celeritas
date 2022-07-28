//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.cc
//---------------------------------------------------------------------------//
#include "GeantImporter.hh"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <G4EmParameters.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4MaterialTable.hh>
#include <G4ParticleTable.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessVector.hh>
#include <G4ProductionCutsTable.hh>
#include <G4RToEConvForElectron.hh>
#include <G4RToEConvForGamma.hh>
#include <G4RToEConvForPositron.hh>
#include <G4RToEConvForProton.hh>
#include <G4SystemOfUnits.hh>
#include <G4Transportation.hh>
#include <G4VSolid.hh>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportParticle.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "detail/GeantExceptionHandler.hh"
#include "detail/GeantLoggerAdapter.hh"
#include "detail/ImportProcessConverter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
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
            return pdg::gamma().get();
        case idxG4ElectronCut:
            return pdg::electron().get();
        case idxG4PositronCut:
            return pdg::positron().get();
        case idxG4ProtonCut:
            return pdg::proton().get();
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
    for (const auto i : range(logical_volume.GetNoDaughters()))
    {
        loop_volumes(volids_volumes,
                     *logical_volume.GetDaughter(i)->GetLogicalVolume());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportParticle vector.
 */
std::vector<ImportParticle> store_particles()
{
    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    std::vector<ImportParticle> particles;

    while (particle_iterator())
    {
        const G4ParticleDefinition& g4_particle_def
            = *(particle_iterator.value());

        PDGNumber pdg{g4_particle_def.GetPDGEncoding()};
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
    }
    CELER_LOG(debug) << "Loaded " << particles.size() << " particles";
    return particles;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportElement vector.
 */
std::vector<ImportElement> store_elements()
{
    const auto& g4element_table = *G4Element::GetElementTable();

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
    CELER_LOG(debug) << "Loaded " << elements.size() << " elements";
    return elements;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportMaterial vector.
 */
std::vector<ImportMaterial> store_materials()
{
    const auto& g4production_cuts_table
        = *G4ProductionCutsTable::GetProductionCutsTable();

    std::vector<ImportMaterial> materials;
    materials.resize(g4production_cuts_table.GetTableSize());

    // Loop over material data
    for (int i : range(g4production_cuts_table.GetTableSize()))
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
        for (int i : range(NumberOfG4CutIndex))
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
        for (int j : range(g4elements->size()))
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

    CELER_LOG(debug) << "Loaded " << materials.size() << " materials";
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportProcess vector.
 */
std::vector<ImportProcess> store_processes()
{
    std::vector<ImportProcess>     processes;
    detail::ImportProcessConverter load_process(
        detail::TableSelection::minimal, store_materials(), store_elements());

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    // XXX To reduce ROOT file data size in repo, only export processes for
    // electron/positron/gamma for now. Allow this as user input later.
    auto include_process = [](PDGNumber pdgnum) -> bool {
        return pdgnum == pdg::electron() || pdgnum == pdg::positron()
               || pdgnum == pdg::gamma();
    };

    while (particle_iterator())
    {
        const G4ParticleDefinition& g4_particle_def
            = *(particle_iterator.value());

        PDGNumber pdgnum(g4_particle_def.GetPDGEncoding());
        if (!pdgnum)
        {
            // Skip "dummy" particles: generic ion and geantino
            continue;
        }

        if (!include_process(pdgnum))
        {
            continue;
        }

        const G4ProcessVector& process_list
            = *g4_particle_def.GetProcessManager()->GetProcessList();

        for (auto j : range(process_list.size()))
        {
            if (dynamic_cast<const G4Transportation*>(process_list[j]))
            {
                // Skip transportation process
                continue;
            }

            if (ImportProcess process
                = load_process(g4_particle_def, *process_list[j]))
            {
                // Not an empty process, so it was not added in a previous loop
                processes.push_back(std::move(process));
            }
        }
    }
    CELER_LOG(debug) << "Loaded " << processes.size() << " processes";
    return processes;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportVolume vector.
 */
std::vector<ImportVolume> store_volumes(const G4VPhysicalVolume* world_volume)
{
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

    CELER_LOG(debug) << "Loaded " << volumes.size() << " volumes";
    return volumes;
}

//---------------------------------------------------------------------------//
/*!
 * Return a \c ImportData::ImportEmParamsMap .
 */
ImportData::ImportEmParamsMap store_em_parameters()
{
    using IEP = ImportEmParameter;

    const auto& g4_em_params = *G4EmParameters::Instance();

    ImportData::ImportEmParamsMap import_em_params{
        {IEP::energy_loss_fluct, g4_em_params.LossFluctuation()},
        {IEP::lpm, g4_em_params.LPM()},
        {IEP::integral_approach, g4_em_params.Integral()},
        {IEP::linear_loss_limit, g4_em_params.LinearLossLimit()},
        {IEP::bins_per_decade, g4_em_params.NumberOfBinsPerDecade()},
        {IEP::min_table_energy, g4_em_params.MinKinEnergy() / MeV},
        {IEP::max_table_energy, g4_em_params.MaxKinEnergy() / MeV},
    };

    CELER_LOG(debug) << "Loaded " << import_em_params.size()
                     << " EM parameters";
    return import_em_params;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from an existing Geant4 geometry, assuming physics is loaded.
 */
GeantImporter::GeantImporter(const G4VPhysicalVolume* world) : world_(world)
{
    CELER_EXPECT(world_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct by capturing a GeantSetup object.
 */
GeantImporter::GeantImporter(GeantSetup&& setup) : setup_(std::move(setup))
{
    CELER_EXPECT(setup_);
    world_ = setup_.world();
    CELER_ENSURE(world_);
}

//---------------------------------------------------------------------------//
/*!
 * Load data from Geant4.
 */
ImportData GeantImporter::operator()(const DataSelection&)
{
    ImportData import_data;
    import_data.particles = store_particles();
    import_data.elements  = store_elements();
    import_data.materials = store_materials();
    import_data.processes = store_processes();
    import_data.volumes   = store_volumes(world_);
    import_data.em_params = store_em_parameters();

    CELER_ENSURE(import_data);
    return import_data;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

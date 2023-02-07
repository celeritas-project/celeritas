//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.cc
//---------------------------------------------------------------------------//
#include "GeantImporter.hh"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Element.hh>
#include <G4ElementTable.hh>
#include <G4ElementVector.hh>
#include <G4EmParameters.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4Navigator.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessType.hh>
#include <G4ProcessVector.hh>
#include <G4ProductionCuts.hh>
#include <G4ProductionCutsTable.hh>
#include <G4RToEConvForElectron.hh>
#include <G4RToEConvForGamma.hh>
#include <G4RToEConvForPositron.hh>
#include <G4RToEConvForProton.hh>
#include <G4String.hh>
#include <G4TransportationManager.hh>
#include <G4Types.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VProcess.hh>
#include <G4VRangeToEnergyConverter.hh>
#include <G4VSolid.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/io/AtomicRelaxationReader.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/LivermorePEReader.hh"
#include "celeritas/io/SeltzerBergerReader.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "detail/AllElementReader.hh"
#include "detail/GeantProcessImporter.hh"

using CLHEP::cm;
using CLHEP::cm2;
using CLHEP::cm3;
using CLHEP::g;
using CLHEP::MeV;
using CLHEP::s;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
decltype(auto) em_particles()
{
    static const std::unordered_set<PDGNumber> particles = {pdg::electron(),
                                                            pdg::positron(),
                                                            pdg::gamma(),
                                                            pdg::mu_minus(),
                                                            pdg::mu_plus()};
    return particles;
}

//---------------------------------------------------------------------------//
//! Filter for desired particle types
struct ParticleFilter
{
    using DataSelection = celeritas::GeantImporter::DataSelection;

    DataSelection::Flags which;

    bool operator()(PDGNumber pdgnum)
    {
        if (!pdgnum)
        {
            return (which & DataSelection::dummy);
        }
        else if (em_particles().count(pdgnum))
        {
            return (which & DataSelection::em);
        }
        else
        {
            // XXX assume non-dummy and non-em are hadronic?
            return (which & DataSelection::hadron);
        }
    }
};

//---------------------------------------------------------------------------//
//! Filter for desired processes
struct ProcessFilter
{
    using DataSelection = celeritas::GeantImporter::DataSelection;

    DataSelection::Flags which;

    bool operator()(G4ProcessType pt)
    {
        switch (pt)
        {
            case G4ProcessType::fElectromagnetic:
                return (which & DataSelection::em);
            case G4ProcessType::fHadronic:
                return (which & DataSelection::hadron);
            default:
                return false;
        }
    }
};

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4State [G4Material.hh] to ImportMaterialState.
 */
ImportMaterialState to_material_state(G4State const& g4_material_state)
{
    switch (g4_material_state)
    {
        case G4State::kStateUndefined:
            return ImportMaterialState::other;
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
PDGNumber to_pdg(G4ProductionCutsIndex const& index)
{
    switch (index)
    {
        case idxG4GammaCut:
            return pdg::gamma();
        case idxG4ElectronCut:
            return pdg::electron();
        case idxG4PositronCut:
            return pdg::positron();
        case idxG4ProtonCut:
            return pdg::proton();
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
void loop_volumes(std::map<int, ImportVolume>& volids_volumes,
                  G4LogicalVolume const& logical_volume)
{
    auto&& [iter, inserted] = volids_volumes.emplace(
        logical_volume.GetInstanceID(), ImportVolume{});
    if (!inserted)
    {
        // Logical volume is already in the map
        return;
    }

    CELER_ASSERT(iter->first >= 0);

    // Fill volume properties
    ImportVolume& volume = iter->second;
    volume.material_id = logical_volume.GetMaterialCutsCouple()->GetIndex();
    volume.name = logical_volume.GetName();
    volume.solid_name = logical_volume.GetSolid()->GetName();

    if (volume.name.empty())
    {
        CELER_LOG(warning)
            << "No logical volume name specified for instance ID "
            << iter->first << " (material " << volume.material_id << ")";
    }

    // Recursive: repeat for every daughter volume, if there are any
    for (auto const i : range(logical_volume.GetNoDaughters()))
    {
        loop_volumes(volids_volumes,
                     *logical_volume.GetDaughter(i)->GetLogicalVolume());
    }

    CELER_ENSURE(volume);
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportParticle vector.
 */
std::vector<ImportParticle>
store_particles(GeantImporter::DataSelection::Flags particle_flags)
{
    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    std::vector<ImportParticle> particles;

    ParticleFilter include_particle{particle_flags};
    while (particle_iterator())
    {
        G4ParticleDefinition const& g4_particle_def
            = *(particle_iterator.value());

        PDGNumber pdg{g4_particle_def.GetPDGEncoding()};
        if (!include_particle(pdg))
        {
            continue;
        }

        ImportParticle particle;
        particle.name = g4_particle_def.GetParticleName();
        particle.pdg = pdg.unchecked_get();
        particle.mass = g4_particle_def.GetPDGMass();
        particle.charge = g4_particle_def.GetPDGCharge();
        particle.spin = g4_particle_def.GetPDGSpin();
        particle.lifetime = g4_particle_def.GetPDGLifeTime();
        particle.is_stable = g4_particle_def.GetPDGStable();

        if (!particle.is_stable)
        {
            // Convert lifetime of unstable particles to seconds
            particle.lifetime /= s;
        }

        particles.push_back(particle);
    }
    CELER_LOG(debug) << "Loaded " << particles.size() << " particles";
    CELER_ENSURE(!particles.empty());
    return particles;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportElement vector.
 */
std::vector<ImportElement> store_elements()
{
    auto const& g4element_table = *G4Element::GetElementTable();

    std::vector<ImportElement> elements;
    elements.resize(g4element_table.size());

    // Loop over element data
    for (auto const& g4element : g4element_table)
    {
        CELER_ASSERT(g4element);

        // Add element to vector
        ImportElement element;
        element.name = g4element->GetName();
        element.atomic_number = g4element->GetZ();
        element.atomic_mass = g4element->GetAtomicMassAmu();
        element.radiation_length_tsai = g4element->GetfRadTsai() / (g / cm2);
        element.coulomb_factor = g4element->GetfCoulomb();

        elements[g4element->GetIndex()] = element;
    }
    CELER_LOG(debug) << "Loaded " << elements.size() << " elements";
    CELER_ENSURE(!elements.empty());
    return elements;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportMaterial vector.
 *
 * TODO: there seems to be an inconsitency between "materials" (index in the
 * global material table) and "material cut couple" (which is what we're
 * defining here?) Maybe we need another level of indirection for material and
 * material+cutoff values?
 */
std::vector<ImportMaterial>
store_materials(GeantImporter::DataSelection::Flags particle_flags)
{
    ParticleFilter include_particle{particle_flags};
    auto const& g4production_cuts_table
        = *G4ProductionCutsTable::GetProductionCutsTable();

    std::vector<ImportMaterial> materials;
    materials.resize(g4production_cuts_table.GetTableSize());
    CELER_VALIDATE(!materials.empty(),
                   << "no Geant4 production cuts are defined (you may need "
                      "to call G4RunManager::RunInitialization)");

    using CutRange = std::pair<G4ProductionCutsIndex,
                               std::unique_ptr<G4VRangeToEnergyConverter>>;

    std::vector<CutRange> cut_converters;
    for (auto gi : range(NumberOfG4CutIndex))
    {
        PDGNumber pdg = to_pdg(gi);
        if (!include_particle(pdg))
        {
            continue;
        }

        std::unique_ptr<G4VRangeToEnergyConverter> converter;
        switch (gi)
        {
            case idxG4GammaCut:
                converter = std::make_unique<G4RToEConvForGamma>();
                break;
            case idxG4ElectronCut:
                converter = std::make_unique<G4RToEConvForElectron>();
                break;
            case idxG4PositronCut:
                converter = std::make_unique<G4RToEConvForPositron>();
                break;
            case idxG4ProtonCut:
                converter = std::make_unique<G4RToEConvForProton>();
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }

        cut_converters.emplace_back(gi, std::move(converter));
    }

    // Loop over material data
    for (auto i : range(materials.size()))
    {
        // Fetch material, element, and production cuts lists
        auto const* g4material_cuts_couple
            = g4production_cuts_table.GetMaterialCutsCouple(i);
        auto const* g4material = g4material_cuts_couple->GetMaterial();
        auto const* g4elements = g4material->GetElementVector();
        auto const* g4prod_cuts = g4material_cuts_couple->GetProductionCuts();

        CELER_ASSERT(g4material_cuts_couple);
        CELER_ASSERT(g4material);
        CELER_ASSERT(g4elements);
        CELER_ASSERT(g4prod_cuts);
        CELER_ASSERT(
            static_cast<std::size_t>(g4material_cuts_couple->GetIndex()) == i);

        // Populate material information
        ImportMaterial material;
        material.name = g4material->GetName();
        material.state = to_material_state(g4material->GetState());
        material.temperature = g4material->GetTemperature();  // [K]
        material.density = g4material->GetDensity() / (g / cm3);
        material.electron_density = g4material->GetTotNbOfElectPerVolume()
                                    / (1. / cm3);
        material.number_density = g4material->GetTotNbOfAtomsPerVolume()
                                  / (1. / cm3);
        material.radiation_length = g4material->GetRadlen() / cm;
        material.nuclear_int_length = g4material->GetNuclearInterLength() / cm;

        // Populate material production cut values
        for (auto const& idx_convert : cut_converters)
        {
            G4ProductionCutsIndex g4i = idx_convert.first;
            G4VRangeToEnergyConverter& converter = *idx_convert.second;

            double const range = g4prod_cuts->GetProductionCut(g4i);
            double const energy = converter.Convert(range, g4material);

            ImportProductionCut cutoffs;
            cutoffs.energy = energy / MeV;
            cutoffs.range = range / cm;

            material.pdg_cutoffs.insert({to_pdg(g4i).get(), cutoffs});
        }

        // Populate element information for this material
        for (int j : range(g4elements->size()))
        {
            auto const& g4element = g4elements->at(j);
            CELER_ASSERT(g4element);

            ImportMatElemComponent elem_comp;
            elem_comp.element_id = g4element->GetIndex();
            elem_comp.mass_fraction = g4material->GetFractionVector()[j];
            double elem_num_density = g4material->GetVecNbOfAtomsPerVolume()[j]
                                      / (1. / cm3);
            elem_comp.number_fraction = elem_num_density
                                        / material.number_density;

            // Add material's element information
            material.elements.push_back(elem_comp);
        }

        // Sort element components by increasing element ID
        std::sort(material.elements.begin(),
                  material.elements.end(),
                  [](ImportMatElemComponent const& lhs,
                     ImportMatElemComponent const& rhs) {
                      return lhs.element_id < rhs.element_id;
                  });

        // Add material to vector
        materials[i] = material;
    }

    CELER_LOG(debug) << "Loaded " << materials.size() << " materials";
    CELER_ENSURE(!materials.empty());
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportProcess vector.
 */
auto store_processes(GeantImporter::DataSelection::Flags process_flags,
                     std::vector<ImportParticle> const& particles,
                     std::vector<ImportElement> const& elements,
                     std::vector<ImportMaterial> const& materials)
    -> std::pair<std::vector<ImportProcess>, std::vector<ImportMscModel>>
{
    ProcessFilter include_process{process_flags};

    std::vector<ImportProcess> processes;
    std::vector<ImportMscModel> msc_models;

    detail::GeantProcessImporter load_process(
        detail::TableSelection::minimal, materials, elements);

    for (auto const& p : particles)
    {
        G4ParticleDefinition const* g4_particle_def
            = G4ParticleTable::GetParticleTable()->FindParticle(p.pdg);
        CELER_ASSERT(g4_particle_def);

        G4ProcessVector const& process_list
            = *g4_particle_def->GetProcessManager()->GetProcessList();

        for (auto j : range(process_list.size()))
        {
            G4VProcess const& process = *process_list[j];
            if (!include_process(process.GetProcessType()))
            {
                CELER_LOG(debug)
                    << "Filtered process '" << process.GetProcessName() << "'";
                continue;
            }

            if (ImportProcess ip = load_process(*g4_particle_def, process))
            {
                // Not an empty process, so it was not added in a previous loop
                if (ip.process_class != ImportProcessClass::msc)
                {
                    processes.push_back(std::move(ip));
                }
                else
                {
                    // Unfold process to MSC models
                    CELER_ASSERT(ip.models.size() == ip.tables.size());
                    for (auto i : range(ip.models.size()))
                    {
                        CELER_ASSERT(ip.tables[i].table_type
                                     == ImportTableType::lambda);
                        ImportMscModel imm;
                        imm.particle_pdg = ip.particle_pdg;
                        imm.model_class = ip.models[i].model_class;
                        imm.lambda_table = std::move(ip.tables[i]);
                        msc_models.push_back(std::move(imm));
                    }
                }
            }
        }
    }
    CELER_LOG(debug) << "Loaded " << processes.size() << " processes";
    return {std::move(processes), std::move(msc_models)};
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportVolume vector.
 */
std::vector<ImportVolume> store_volumes(G4VPhysicalVolume const* world_volume)
{
    std::vector<ImportVolume> volumes;
    std::map<int, ImportVolume> volids_volumes;

    // Recursive loop over all logical volumes to populate map<volid, volume>
    loop_volumes(volids_volumes, *world_volume->GetLogicalVolume());

    // Populate vector<ImportVolume>
    volumes.resize(volids_volumes.size());
    for (auto&& [volid, volume] : volids_volumes)
    {
        if (static_cast<std::size_t>(volid) >= volumes.size())
        {
            volumes.resize(volid + 1);
        }
        volumes[volid] = std::move(volume);
    }

    CELER_LOG(debug) << "Loaded " << volumes.size() << " volumes";
    return volumes;
}

//---------------------------------------------------------------------------//
/*!
 * Return a \c ImportData::ImportEmParamsMap .
 */
ImportEmParameters store_em_parameters()
{
    ImportEmParameters import;

    auto const& g4 = *G4EmParameters::Instance();

    import.energy_loss_fluct = g4.LossFluctuation();
    import.lpm = g4.LPM();
    import.integral_approach = g4.Integral();
    import.linear_loss_limit = g4.LinearLossLimit();
    import.auger = g4.Auger();

    CELER_ENSURE(import);
    return import;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get an externally loaded Geant4 top-level geometry element.
 *
 * This is only defined if Geant4 has already been set up. It's meant to be
 * used in concert with GeantImporter or other Geant-importing classes.
 */
G4VPhysicalVolume const* GeantImporter::get_world_volume()
{
    auto* man = G4TransportationManager::GetTransportationManager();
    CELER_ASSERT(man);
    auto* nav = man->GetNavigatorForTracking();
    CELER_ASSERT(nav);
    auto* world = nav->GetWorldVolume();
    CELER_ENSURE(world);
    return world;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from an existing Geant4 geometry, assuming physics is loaded.
 */
GeantImporter::GeantImporter(G4VPhysicalVolume const* world) : world_(world)
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
ImportData GeantImporter::operator()(DataSelection const& selected)
{
    ImportData import_data;

    {
        CELER_LOG(status) << "Transferring data from Geant4";
        ScopedTimeLog scoped_time;
        import_data.particles = store_particles(selected.particles);
        import_data.elements = store_elements();
        import_data.materials = store_materials(selected.particles);
        // TODO: when moving to C++17, use a structured binding
        auto processes_and_msc = store_processes(selected.processes,
                                                 import_data.particles,
                                                 import_data.elements,
                                                 import_data.materials);
        import_data.processes = std::move(processes_and_msc.first);
        import_data.msc_models = std::move(processes_and_msc.second);
        import_data.volumes = store_volumes(world_);
        if (selected.processes & DataSelection::em)
        {
            import_data.em_params = store_em_parameters();
        }
    }

    if (selected.reader_data)
    {
        CELER_LOG(status) << "Loading external elemental data";
        ScopedTimeLog scoped_time;

        detail::AllElementReader load_data{import_data.elements};

        auto have_process = [&import_data](ImportProcessClass ipc) {
            return std::any_of(import_data.processes.begin(),
                               import_data.processes.end(),
                               [ipc](const ImportProcess& ip) {
                                   return ip.process_class == ipc;
                               });
        };

        if (have_process(ImportProcessClass::e_brems))
        {
            import_data.sb_data = load_data(SeltzerBergerReader{});
        }
        if (have_process(ImportProcessClass::photoelectric))
        {
            import_data.livermore_pe_data = load_data(LivermorePEReader{});
        }
        if (G4EmParameters::Instance()->Fluo())
        {
            // TODO: only read auger data if that option is enabled
            import_data.atomic_relaxation_data
                = load_data(AtomicRelaxationReader{});
        }
        else if (G4EmParameters::Instance()->Auger())
        {
            CELER_LOG(warning) << "Auger emission is ignored because "
                                  "fluorescent atomic relaxation is disabled";
        }
    }

    CELER_ENSURE(import_data);
    return import_data;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

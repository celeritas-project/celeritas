//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.cc
//---------------------------------------------------------------------------//
#include "GeantImporter.hh"

#include <algorithm>
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
#include <G4GammaGeneralProcess.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Material.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4MscStepLimitType.hh>
#include <G4MuonMinusAtomicCapture.hh>
#include <G4Navigator.hh>
#include <G4NuclearFormfactorType.hh>
#include <G4NucleiProperties.hh>
#include <G4OpAbsorption.hh>
#include <G4OpBoundaryProcess.hh>
#include <G4OpMieHG.hh>
#include <G4OpRayleigh.hh>
#include <G4OpWLS.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessType.hh>
#include <G4ProcessVector.hh>
#include <G4ProductionCuts.hh>
#include <G4ProductionCutsTable.hh>
#include <G4PropagatorInField.hh>
#include <G4RToEConvForElectron.hh>
#include <G4RToEConvForGamma.hh>
#include <G4RToEConvForPositron.hh>
#include <G4RToEConvForProton.hh>
#include <G4Region.hh>
#include <G4RegionStore.hh>
#include <G4Scintillation.hh>
#include <G4String.hh>
#include <G4Transportation.hh>
#include <G4Types.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4VMultipleScattering.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VProcess.hh>
#include <G4VRangeToEnergyConverter.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/inp/Grid.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportUnits.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "GeantParticleView.hh"
#include "GeantSetup.hh"

#include "detail/GeantMaterialPropertyGetter.hh"
#include "detail/GeantPhysicsLoader.hh"
#include "detail/GeantProcessImporter.hh"

inline constexpr double mev_scale = 1 / CLHEP::MeV;
inline constexpr celeritas::PDGNumber g4_optical_pdg{-22};

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
decltype(auto) em_basic_particles()
{
    static std::unordered_set<PDGNumber> const particles
        = {pdg::electron(), pdg::positron(), pdg::gamma()};
    return particles;
}

//---------------------------------------------------------------------------//
decltype(auto) em_ex_particles()
{
    static std::unordered_set<PDGNumber> const particles
        = {pdg::mu_minus(), pdg::mu_plus()};
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
        else if (em_basic_particles().count(pdgnum))
        {
            return (which & DataSelection::em_basic);
        }
        else if (em_ex_particles().count(pdgnum))
        {
            return (which & DataSelection::em_ex);
        }
        else if (pdgnum == g4_optical_pdg)
        {
            return (which & DataSelection::optical);
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
    using EmSubType = G4EmProcessSubType;

    DataSelection::Flags which;

    bool operator()(G4ProcessType pt, int subtype)
    {
        switch (pt)
        {
            case G4ProcessType::fElectromagnetic:
                if (which & DataSelection::em)
                {
                    return true;
                }
                else if (which & DataSelection::optical)
                {
                    // Also allow the process if it's an optical generation
                    // process and the user has allowed optical process types
                    auto emst = static_cast<EmSubType>(subtype);
                    return emst == EmSubType::fScintillation
                           || emst == EmSubType::fCerenkov;
                }
                else
                {
                    return false;
                }
            case G4ProcessType::fOptical:
                return (which & DataSelection::optical);
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
 * Safely switch from G4MscStepLimitType [G4MscStepLimitType.hh] to
 * MscStepLimitAlgorithm.
 */
MscStepLimitAlgorithm
to_msc_step_algorithm(G4MscStepLimitType const& msc_step_algorithm)
{
    switch (msc_step_algorithm)
    {
        case G4MscStepLimitType::fMinimal:
            return MscStepLimitAlgorithm::minimal;
        case G4MscStepLimitType::fUseSafety:
            return MscStepLimitAlgorithm::safety;
        case G4MscStepLimitType::fUseSafetyPlus:
            return MscStepLimitAlgorithm::safety_plus;
        case G4MscStepLimitType::fUseDistanceToBoundary:
            return MscStepLimitAlgorithm::distance_to_boundary;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Safely switch from G4NuclearFormfactorType [G4NuclearFormfactorType.hh] to
 * NuclearFormFactorType.
 */
NuclearFormFactorType
to_form_factor_type(G4NuclearFormfactorType const& form_factor_type)
{
    switch (form_factor_type)
    {
        case G4NuclearFormfactorType::fNoneNF:
            return NuclearFormFactorType::none;
        case G4NuclearFormfactorType::fExponentialNF:
            return NuclearFormFactorType::exponential;
        case G4NuclearFormfactorType::fGaussianNF:
            return NuclearFormFactorType::gaussian;
        case G4NuclearFormfactorType::fFlatNF:
            return NuclearFormFactorType::flat;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c inp::Particle vector.
 */
std::vector<inp::Particle>
import_particles(GeantImporter::DataSelection::Flags particle_flags)
{
    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    std::vector<inp::Particle> particles;

    ParticleFilter include_particle{particle_flags};
    while (particle_iterator())
    {
        GeantParticleView particle_view{*(particle_iterator.value())};

        PDGNumber pdg = [&] {
            if (G4VERSION_NUMBER < 1070 && particle_view.is_optical_photon())
            {
                // Before 10.7, geant4 uses PDG 0 plus a unique string name
                return g4_optical_pdg;
            }
            return particle_view.pdg();
        }();

        if (!include_particle(pdg))
        {
            continue;
        }

        particles.push_back([&particle_view, &pdg] {
            inp::Particle result;
            result.name = particle_view.name();
            result.pdg = pdg;
            result.mass = particle_view.mass();
            result.charge = particle_view.charge();
            result.decay_constant
                = native_value_from(particle_view.decay_constant());
            return result;
        }());
    }

    CELER_LOG(debug) << "Loaded " << particles.size() << " particles";
    CELER_ENSURE(!particles.empty());
    return particles;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportIsotope vector.
 */
std::vector<ImportIsotope> import_isotopes()
{
    auto const& g4isotope_table = *G4Isotope::GetIsotopeTable();
    CELER_EXPECT(!g4isotope_table.empty());

    std::vector<ImportIsotope> isotopes(g4isotope_table.size());
    for (auto idx : range(g4isotope_table.size()))
    {
        if (!g4isotope_table[idx])
        {
            CELER_LOG(warning) << "Skipping import of null isotope at index \'"
                               << idx << "\' of the G4IsotopeTable";
            continue;
        }
        auto const& g4isotope = *g4isotope_table[idx];

        ImportIsotope& isotope = isotopes[idx];
        isotope.name = g4isotope.GetName();
        isotope.atomic_number = g4isotope.GetZ();
        isotope.atomic_mass_number = g4isotope.GetN();
        isotope.binding_energy = G4NucleiProperties::GetBindingEnergy(
            isotope.atomic_mass_number, isotope.atomic_number);

        // Binding energy difference for losing a nucleon
        if (isotope.atomic_mass_number > 1 && isotope.atomic_number > 1
            && isotope.atomic_mass_number >= isotope.atomic_number)
        {
            isotope.proton_loss_energy
                = G4NucleiProperties::GetBindingEnergy(
                      isotope.atomic_mass_number, isotope.atomic_number)
                  - G4NucleiProperties::GetBindingEnergy(
                      isotope.atomic_mass_number - 1,
                      isotope.atomic_number - 1);
            isotope.neutron_loss_energy
                = G4NucleiProperties::GetBindingEnergy(
                      isotope.atomic_mass_number, isotope.atomic_number)
                  - G4NucleiProperties::GetBindingEnergy(
                      isotope.atomic_mass_number - 1, isotope.atomic_number);
        }
        else
        {
            isotope.proton_loss_energy = 0;
            isotope.neutron_loss_energy = 0;
        }

        isotope.nuclear_mass = G4NucleiProperties::GetNuclearMass(
            isotope.atomic_mass_number, isotope.atomic_number);
    }

    CELER_ENSURE(!isotopes.empty());
    CELER_LOG(debug) << "Loaded " << isotopes.size() << " isotopes";
    return isotopes;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportElement vector.
 */
std::vector<ImportElement> import_elements()
{
    std::vector<ImportElement> elements;

    auto const& g4element_table = *G4Element::GetElementTable();
    CELER_EXPECT(!g4element_table.empty());

    elements.resize(g4element_table.size());

    // Loop over element data
    for (auto const& g4element : g4element_table)
    {
        CELER_ASSERT(g4element);
        auto const& g4isotope_vec = *g4element->GetIsotopeVector();
        CELER_ASSERT(g4isotope_vec.size() == g4element->GetNumberOfIsotopes());

        // Add element to ImportElement vector
        ImportElement element;
        element.name = g4element->GetName();
        element.atomic_number = g4element->GetZ();
        element.atomic_mass = g4element->GetAtomicMassAmu();

        // Despite the function name, this is *NOT* a vector, it's an array
        double* const g4rel_abundance = g4element->GetRelativeAbundanceVector();

        double total_el_abundance_fraction = 0;  // Verify that the sum is ~1
        for (auto idx : range(g4element->GetNumberOfIsotopes()))
        {
            ImportElement::IsotopeFrac key;
            key.first = g4isotope_vec[idx]->GetIndex();
            key.second = g4rel_abundance[idx];
            element.isotopes_fractions.push_back(std::move(key));

            total_el_abundance_fraction += g4rel_abundance[idx];
        }
        CELER_VALIDATE(soft_equal(1., total_el_abundance_fraction),
                       << "Total relative isotopic abundance for element `"
                       << element.name
                       << "` should sum to 1, but instead sum to "
                       << total_el_abundance_fraction);

        elements[g4element->GetIndex()] = element;
    }

    CELER_ENSURE(!elements.empty());
    CELER_LOG(debug) << "Loaded " << elements.size() << " elements";
    return elements;
}

//---------------------------------------------------------------------------//
/*!
 * Store material-dependent optical properties.
 *
 * This returns a vector of optical materials corresponding to an "optical
 * material ID".
 */
std::vector<ImportOpticalMaterial>
import_optical_materials(GeoOpticalIdMap const& geo_to_opt)
{
    if (geo_to_opt.empty())
    {
        CELER_LOG(warning)
            << R"(Optical materials were requested but none are present)";
        // No optical materials
        return {};
    }

    auto const& mt = *G4Material::GetMaterialTable();
    CELER_ASSERT(mt.size() == geo_to_opt.num_geo());

    std::vector<ImportOpticalMaterial> result(geo_to_opt.num_optical());

    // Loop over optical materials
    for (auto geo_mat_id : range(GeoMatId{geo_to_opt.num_geo()}))
    {
        auto opt_mat_id = geo_to_opt[geo_mat_id];
        if (!opt_mat_id)
        {
            continue;
        }
        // Get Geant4 material properties
        G4Material const* material = mt[geo_mat_id.get()];
        CELER_ASSERT(material);
        CELER_ASSERT(geo_mat_id == id_cast<GeoMatId>(material->GetIndex()));
        detail::GeantMaterialPropertyGetter get_property{
            material->GetMaterialPropertiesTable(), material->GetName()};

        // Optical materials should map uniquely
        ImportOpticalMaterial& optical = result[opt_mat_id.get()];
        CELER_ASSERT(!optical);

        // Save common properties
        bool has_rindex
            = get_property(optical.properties.refractive_index,
                           "RINDEX",
                           {ImportUnits::mev, ImportUnits::unitless});
        // Existence of RINDEX should correspond to GeoOpticalIdMap
        // construction
        CELER_ASSERT(has_rindex);

        // Most properties are loaded by GeantPhysicsLoader:
        // Scintillation, WLS, WLS2, Mie

        CELER_VALIDATE(optical,
                       << "failed to load valid optical material data for "
                          "OptMatId{"
                       << opt_mat_id.get() << "} = " << material->GetName());
    }

    CELER_LOG(debug) << "Loaded " << result.size() << " optical materials";
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportGeoMaterial vector.
 *
 * These are the ground-truth physical properties of the materials with no
 * information about how user physics selections/options affect
 * production cutoffs etc.
 */
std::vector<ImportGeoMaterial> import_geo_materials()
{
    auto const& mt = *G4Material::GetMaterialTable();

    std::vector<ImportGeoMaterial> materials;
    materials.resize(mt.size());
    CELER_VALIDATE(!materials.empty(), << "no Geant4 materials are defined");

    double const numdens_scale
        = native_value_from_clhep(ImportUnits::inv_len_cb);

    // Loop over material data
    for (auto i : range(materials.size()))
    {
        auto const* g4material = mt[i];
        CELER_ASSERT(g4material);
        CELER_ASSERT(i == static_cast<std::size_t>(g4material->GetIndex()));
        auto const* g4elements = g4material->GetElementVector();
        CELER_ASSERT(g4elements);

        // Populate material information
        ImportGeoMaterial material;
        material.name = g4material->GetName();
        material.state = to_material_state(g4material->GetState());
        material.temperature = g4material->GetTemperature();  // [K]
        material.number_density = g4material->GetTotNbOfAtomsPerVolume()
                                  * numdens_scale;

        // Populate element information for this material
        for (int j : range(g4elements->size()))
        {
            auto const& g4element = g4elements->at(j);
            CELER_ASSERT(g4element);

            ImportMatElemComponent elem_comp;
            elem_comp.element_id = g4element->GetIndex();
            double elem_num_density = g4material->GetVecNbOfAtomsPerVolume()[j]
                                      * numdens_scale;
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
        materials[i] = std::move(material);
    }

    CELER_LOG(debug) << "Loaded " << materials.size() << " geo materials";
    CELER_ENSURE(!materials.empty());
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportPhysMaterial vector.
 */
std::vector<ImportPhysMaterial>
import_phys_materials(GeantImporter::DataSelection::Flags particle_flags,
                      GeoOpticalIdMap const& geo_to_opt)
{
    ParticleFilter include_particle{particle_flags};
    auto const& pct = *G4ProductionCutsTable::GetProductionCutsTable();

    std::vector<ImportPhysMaterial> materials;
    materials.resize(pct.GetTableSize());
    CELER_VALIDATE(!materials.empty(),
                   << "no Geant4 production cuts are defined (you may "
                      "need to call G4RunManager::RunInitialization)");

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

    double const len_scale = native_value_from_clhep(ImportUnits::len);

    // Loop over material data
    for (auto i : range(materials.size()))
    {
        auto const* mcc = pct.GetMaterialCutsCouple(i);
        CELER_ASSERT(mcc);
        CELER_ASSERT(static_cast<std::size_t>(mcc->GetIndex()) == i);

        ImportPhysMaterial material;

        // Save corresponding material IDs
        auto const* g4material = mcc->GetMaterial();
        CELER_ASSERT(g4material);
        material.geo_material_id = g4material->GetIndex();
        if (!geo_to_opt.empty())
        {
            if (auto opt_id
                = geo_to_opt[id_cast<GeoMatId>(g4material->GetIndex())])
            {
                // Assign the optical material corresponding to the geometry
                // material
                material.optical_material_id = opt_id.get();
            }
        }

        // Populate material production cut values
        auto const* g4prod_cuts = mcc->GetProductionCuts();
        CELER_ASSERT(g4prod_cuts);
        for (auto const& idx_convert : cut_converters)
        {
            G4ProductionCutsIndex g4i = idx_convert.first;
            G4VRangeToEnergyConverter& converter = *idx_convert.second;

            double const range = g4prod_cuts->GetProductionCut(g4i);
            double const energy = converter.Convert(range, g4material);

            ImportProductionCut cutoffs;
            cutoffs.energy = energy * mev_scale;
            cutoffs.range = range * len_scale;

            material.pdg_cutoffs.insert({to_pdg(g4i).get(), cutoffs});
        }

        // Add material to vector
        materials[i] = std::move(material);
    }

    CELER_LOG(debug) << "Loaded " << materials.size() << " physics materials";
    CELER_ENSURE(!materials.empty());
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * GEt the process list of a particle.
 */
G4ProcessVector const& get_process_vec(G4ParticleDefinition const& p)
{
    auto const* pm = p.GetProcessManager();
    CELER_VALIDATE(
        pm, << "No process manager for '" << p.GetParticleName() << "'");

    auto* pl = pm->GetProcessList();
    CELER_ENSURE(pl);
    return *pl;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportProcess vector.
 *
 * TODO: instead of looping over all particles, loop over all *offload*
 * particles, i.e. the ones Celeritas is actively stepping through. Warn if
 * any processes that apply to the particle aren't known to/implemented by
 * Celeritas. (Ignoring processes can be done as a post-import step.) From the
 * list of imported processes, we can determine what secondary particle types
 * are being created, and thence to the list of particle definitions to import.
 */
auto import_processes(GeantImporter::DataSelection selected,
                      GeoOpticalIdMap const& geo_to_opt,
                      ImportData& imported)
{
    ParticleFilter include_particle{selected.particles};
    ProcessFilter include_process{selected.processes};

    auto const& particles = imported.particles;
    auto const& elements = imported.elements;
    auto const& materials = imported.phys_materials;

    auto& processes = imported.processes;
    auto& msc_models = imported.msc_models;

    static celeritas::TypeDemangler<G4VProcess> const demangle_process;
    detail::GeantProcessImporter legacy_import_process(
        materials, elements, selected.interpolation);
    detail::GeantPhysicsLoader load_physics(imported, geo_to_opt);

    auto append_process = [&](G4ParticleDefinition const& particle,
                              G4VProcess const& process) -> void {
        if (load_physics(process))
        {
            // We were able to load (or ignore) this specific process
            return;
        }

        // Load per-particle data (runs alongside process-level loader)
        // TODO: should call legacy_import_process if needed, and return early
        // if data is imported
        load_physics(GeantParticleView{particle}, process);

        // TODO: move implementation to GeantPhysicsLoader
        if (auto const* gg_process
            = dynamic_cast<G4GammaGeneralProcess const*>(&process))
        {
#if G4VERSION_NUMBER >= 1060
            // Extract the real EM processes embedded inside "gamma general"
            // using an awkward string-based lookup which is the only one
            // available to us :(
            for (auto emproc_enum : range(ImportProcessClass::size_))
            {
                if (G4VEmProcess const* subprocess
                    = const_cast<G4GammaGeneralProcess*>(gg_process)
                          ->GetEmProcess(to_geant_name(emproc_enum)))
                {
                    processes.push_back(
                        legacy_import_process(particle, *subprocess));
                }
            }
#else
            CELER_DISCARD(gg_process);
            CELER_NOT_IMPLEMENTED("GammaGeneralProcess for Geant4 < 10.6");
#endif
        }
        else if (auto const* em_process
                 = dynamic_cast<G4VEmProcess const*>(&process))
        {
            processes.push_back(legacy_import_process(particle, *em_process));
        }
        else if (auto const* el_process
                 = dynamic_cast<G4VEnergyLossProcess const*>(&process))
        {
            processes.push_back(legacy_import_process(particle, *el_process));
        }
        else if (auto const* msc_process
                 = dynamic_cast<G4VMultipleScattering const*>(&process))
        {
            // Unpack MSC process into multiple MSC models
            auto new_msc_models = legacy_import_process(particle, *msc_process);
            msc_models.insert(msc_models.end(),
                              std::make_move_iterator(new_msc_models.begin()),
                              std::make_move_iterator(new_msc_models.end()));
        }
        else
        {
            CELER_LOG(error)
                << "Cannot load unknown process '" << process.GetProcessName()
                << "' (RTTI: " << demangle_process(process) << ")";
        }
    };

    for (auto const& p : particles)
    {
        G4ParticleDefinition* g4_particle_def;
        if (G4VERSION_NUMBER < 1070 && p.pdg == g4_optical_pdg)
        {
            // Optical photon PDG in Geant4 is 0 before version 10.7
            g4_particle_def = G4OpticalPhoton::OpticalPhoton();
        }
        else
        {
            g4_particle_def = G4ParticleTable::GetParticleTable()->FindParticle(
                p.pdg.get());
        }
        CELER_ASSERT(g4_particle_def);

        if (!include_particle(p.pdg))
        {
            CELER_LOG(debug) << "Filtered all processes from particle '"
                             << g4_particle_def->GetParticleName() << "'";
            continue;
        }

        auto const& process_vec = get_process_vec(*g4_particle_def);

        for (auto j : range(process_vec.size()))
        {
            G4VProcess const& process = *process_vec[j];
            if (!include_process(process.GetProcessType(),
                                 process.GetProcessSubType()))
            {
                continue;
            }

            append_process(*g4_particle_def, process);
        }
    }

    CELER_LOG(debug) << "Loaded " << processes.size() << " EM processes";
    CELER_LOG(debug) << "Loaded " << msc_models.size() << " MSC models";
}

//---------------------------------------------------------------------------//
/*!
 * Get the transportation process for a given particle type.
 */
G4Transportation const& find_transportation(G4ParticleDefinition const& p)
{
    // Search through the processes to find transportion
    auto const& pl = get_process_vec(p);
    for (auto i : range(pl.size()))
    {
        if (auto const* trans = dynamic_cast<G4Transportation const*>(pl[i]))
        {
            return *trans;
        }
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Store particle-dependent transportation parameters.
 */
ImportTransParameters
import_trans_parameters(GeantImporter::DataSelection::Flags particle_flags)
{
    ImportTransParameters result;

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();
    ParticleFilter include_particle{particle_flags};
    while (particle_iterator())
    {
        PDGNumber pdg = [&particle_iterator] {
            GeantParticleView particle{*(particle_iterator.value())};
            if (G4VERSION_NUMBER < 1070 && particle.is_optical_photon())
            {
                // Before 10.7, geant4 uses PDG 0 plus a unique string name
                return g4_optical_pdg;
            }
            return particle.pdg();
        }();

        if (!include_particle(pdg))
        {
            continue;
        }

        // Get the transportation process
        auto const& trans = find_transportation(*particle_iterator.value());
        if (auto const* fp
            = const_cast<G4Transportation&>(trans).GetPropagatorInField())
        {
            result.max_substeps = fp->GetMaxLoopCount();
        }

        // Get the threshold values for killing looping tracks
        ImportLoopingThreshold looping;
        looping.threshold_trials = trans.GetThresholdTrials();
        looping.important_energy = trans.GetThresholdImportantEnergy()
                                   * mev_scale;
        CELER_ASSERT(looping);
        result.looping.insert({pdg.get(), looping});
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return a \c ImportData::ImportEmParamsMap .
 */
ImportEmParameters import_em_parameters()
{
    ImportEmParameters import;

    auto const& g4 = *G4EmParameters::Instance();
    double const len_scale = native_value_from_clhep(ImportUnits::len);

    import.energy_loss_fluct = g4.LossFluctuation();
    import.lpm = g4.LPM();
    import.integral_approach = g4.Integral();
    import.linear_loss_limit = g4.LinearLossLimit();
    import.lowest_electron_energy = g4.LowestElectronEnergy() * mev_scale;
    import.lowest_muhad_energy = g4.LowestMuHadEnergy() * mev_scale;
    import.fluorescence = g4.Fluo();
    import.auger = g4.Auger();
    import.msc_step_algorithm = to_msc_step_algorithm(g4.MscStepLimitType());
    import.msc_muhad_step_algorithm
        = to_msc_step_algorithm(g4.MscMuHadStepLimitType());
    import.msc_displaced = g4.LateralDisplacement();
    import.msc_muhad_displaced = g4.MuHadLateralDisplacement();
    import.msc_range_factor = g4.MscRangeFactor();
    import.msc_muhad_range_factor = g4.MscMuHadRangeFactor();
#if G4VERSION_NUMBER >= 1060
    import.msc_safety_factor = g4.MscSafetyFactor();
    import.msc_lambda_limit = g4.MscLambdaLimit() * len_scale;
#else
    CELER_DISCARD(len_scale);
#endif
    import.msc_theta_limit = g4.MscThetaLimit();
    import.angle_limit_factor = g4.FactorForAngleLimit();
    import.apply_cuts = g4.ApplyCuts();
    import.screening_factor = g4.ScreeningFactor();
    import.form_factor = to_form_factor_type(g4.NuclearFormfactorType());

    CELER_ENSURE(import);
    return import;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportVolume vector.
 */
std::vector<ImportVolume> import_volumes()
{
    auto geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geo, << "global Geant4 geometry is not loaded");

    VolumeParams volume_params{geo->make_model_input().volumes};

    auto const& volumes = volume_params.volume_labels();
    std::vector<ImportVolume> result(volumes.size());
    size_type count{0};

    for (auto vol_id : range(VolumeId{volumes.size()}))
    {
        auto* g4lv = geo->id_to_geant(vol_id);
        if (!g4lv)
            continue;

        ImportVolume& volume = result[vol_id.get()];
        if (auto* mat = g4lv->GetMaterial())
        {
            volume.geo_material_id = mat->GetIndex();
        }
        if (auto* reg = g4lv->GetRegion())
        {
            volume.region_id = reg->GetInstanceID();
        }
        if (auto* cuts = g4lv->GetMaterialCutsCouple())
        {
            volume.phys_material_id = cuts->GetIndex();
        }
        volume.name = to_string(volume_params.volume_labels().at(vol_id));
        volume.solid_name = g4lv->GetSolid()->GetName();

        ++count;
    }

    CELER_LOG(debug) << "Loaded " << count << " of " << result.size()
                     << " volumes";
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from an existing Geant4 geometry, assuming physics is loaded.
 */
GeantImporter::GeantImporter()
{
    CELER_EXPECT(!celeritas::global_geant_geo().expired());
}

//---------------------------------------------------------------------------//
/*!
 * Construct by capturing a GeantSetup object.
 */
GeantImporter::GeantImporter(GeantSetup&& setup) : setup_(std::move(setup))
{
    CELER_EXPECT(setup_);
    CELER_EXPECT(!celeritas::global_geant_geo().expired());
}

//---------------------------------------------------------------------------//
/*!
 * Load data from Geant4.
 */
ImportData GeantImporter::operator()(DataSelection const& selected)
{
    CELER_VALIDATE(
        (selected.materials && selected.particles != DataSelection::none)
            || selected.processes == DataSelection::none,
        << "materials and particles must be enabled if requesting processes");
    ImportData imported;

    {
        CELER_LOG(status) << "Transferring data from Geant4";
        ScopedGeantExceptionHandler scoped_exceptions;
        ScopedTimeLog scoped_time;

        auto geo_to_opt = std::make_shared<GeoOpticalIdMap>();

        if (selected.particles != DataSelection::none)
        {
            imported.particles = import_particles(selected.particles);
        }
        if (selected.materials)
        {
            if (selected.processes & DataSelection::optical)
            {
                auto geo = celeritas::global_geant_geo().lock();
                CELER_VALIDATE(geo, << "global Geant4 geometry is not loaded");

                geo_to_opt = geo->geo_optical_id_map();
                imported.optical_materials
                    = import_optical_materials(*geo_to_opt);
            }

            imported.isotopes = import_isotopes();
            imported.elements = import_elements();
            imported.geo_materials = import_geo_materials();
            imported.phys_materials
                = import_phys_materials(selected.particles, *geo_to_opt);
        }
        if (selected.processes != DataSelection::none)
        {
            import_processes(selected, *geo_to_opt, imported);
        }

        imported.volumes = import_volumes();
        if (selected.particles != DataSelection::none)
        {
            imported.trans_params = import_trans_parameters(selected.particles);
        }
        if (selected.processes & DataSelection::em)
        {
            imported.em_params = import_em_parameters();
            imported.em_params.interpolation = selected.interpolation;
        }
    }

    imported.units = units::NativeTraits::label();
    return imported;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include <tuple>
#include <unordered_map>
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
#include <G4Navigator.hh>
#include <G4NuclearFormfactorType.hh>
#include <G4NucleiProperties.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessType.hh>
#include <G4ProcessVector.hh>
#include <G4ProductionCuts.hh>
#include <G4ProductionCutsTable.hh>
#include <G4PropagatorInField.hh>
#include <G4Region.hh>
#include <G4RegionStore.hh>
#include <G4RToEConvForElectron.hh>
#include <G4RToEConvForGamma.hh>
#include <G4RToEConvForPositron.hh>
#include <G4RToEConvForProton.hh>
#include <G4String.hh>
#include <G4Transportation.hh>
#include <G4TransportationManager.hh>
#include <G4Types.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4Version.hh>
#include <G4VMultipleScattering.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VProcess.hh>
#include <G4VRangeToEnergyConverter.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/g4/VisitGeantVolumes.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "celeritas/io/AtomicRelaxationReader.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/LivermorePEReader.hh"
#include "celeritas/io/SeltzerBergerReader.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "GeantSetup.hh"

#include "detail/AllElementReader.hh"
#include "detail/GeantProcessImporter.hh"

inline constexpr double mev_scale = 1 / CLHEP::MeV;

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
//! Retrieve and store optical material properties, if present.
struct MatPropGetter
{
    using MPT = G4MaterialPropertiesTable;

    MPT const& mpt;

    void scalar(double* dst, char const* name, ImportUnits q)
    {
        if (!mpt.ConstPropertyExists(name))
        {
            return;
        }
        *dst = mpt.GetConstProperty(name) * native_value_from_clhep(q);
    }

    void scalar(double* dst, std::string name, int comp, ImportUnits q)
    {
        // Geant4 10.6 and earlier require a const char* argument
        name += std::to_string(comp);
        this->scalar(dst, name.c_str(), q);
    }

    void
    vector(ImportPhysicsVector* dst, std::string const& name, ImportUnits q)
    {
        // Geant4@10.7: G4MaterialPropertiesTable.GetProperty is not const
        // and <=10.6 require const char*
        auto const* g4vector = const_cast<MPT&>(mpt).GetProperty(name.c_str());
        if (!g4vector)
        {
            return;
        }
        *dst = detail::import_physics_vector(*g4vector, {ImportUnits::mev, q});
    }
};

//---------------------------------------------------------------------------//
//! Map particles defined in \c G4MaterialConstPropertyIndex .
auto& optical_particles_map()
{
    static std::unordered_map<std::string, PDGNumber> const map
        = {{"PROTON", pdg::proton()},
           {"DEUTERON", pdg::deuteron()},
           {"TRITON", pdg::triton()},
           {"ALPHA", pdg::alpha()},
           {"ION", pdg::ion()},
           {"ELECTRON", pdg::electron()}};
    return map;
}

//---------------------------------------------------------------------------//
/*!
 * Populate an \c ImportScintComponent .
 * To retrieve a material-only component simply do not use particle name.
 */
std::vector<ImportScintComponent>
fill_vec_import_scint_comp(MatPropGetter& get_property,
                           std::string particle_name = "")
{
    CELER_EXPECT(particle_name.empty()
                 || optical_particles_map().count(particle_name));

    std::vector<ImportScintComponent> components;
    for (int comp_idx : range(1, 4))
    {
        ImportScintComponent comp;
        get_property.scalar(&comp.yield_per_energy,
                            particle_name + "SCINTILLATIONYIELD",
                            comp_idx,
                            ImportUnits::inv_mev);

        // Custom-defined properties not available in G4MaterialPropertyIndex
        get_property.scalar(&comp.lambda_mean,
                            particle_name + "SCINTILLATIONLAMBDAMEAN",
                            comp_idx,
                            ImportUnits::len);
        get_property.scalar(&comp.lambda_sigma,
                            particle_name + "SCINTILLATIONLAMBDASIGMA",
                            comp_idx,
                            ImportUnits::len);

        // Rise time is not defined for particle type in Geant4
        get_property.scalar(&comp.rise_time,
                            particle_name + "SCINTILLATIONRISETIME",
                            comp_idx,
                            ImportUnits::time);

        get_property.scalar(&comp.fall_time,
                            particle_name + "SCINTILLATIONTIMECONSTANT",
                            comp_idx,
                            ImportUnits::time);
        if (comp)
        {
            components.push_back(std::move(comp));
        }
    }
    return components;
}

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
 * Return a populated \c ImportParticle vector.
 */
std::vector<ImportParticle>
import_particles(GeantImporter::DataSelection::Flags particle_flags)
{
    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();

    std::vector<ImportParticle> particles;

    double const time_scale = native_value_from_clhep(ImportUnits::time);

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
            particle.lifetime *= time_scale;
        }

        particles.push_back(particle);
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
 * This returns a map of material index to imported optical property data.
 */
ImportData::ImportOpticalMap import_optical()
{
    auto const& mt = *G4Material::GetMaterialTable();
    CELER_ASSERT(mt.size() > 0);

    auto const& particle_map = optical_particles_map();
    ImportData::ImportOpticalMap result;

    // Loop over optical materials
    for (auto mat_idx : range(mt.size()))
    {
        G4Material const* material = mt[mat_idx];
        CELER_ASSERT(material);
        CELER_ASSERT(mat_idx == static_cast<std::size_t>(material->GetIndex()));

        // Add optical material properties, if any are present
        auto const* mpt = material->GetMaterialPropertiesTable();
        if (!mpt)
        {
            continue;
        }
        ImportOpticalMaterial optical;
        MatPropGetter get_property{*mpt};

        // Save common properties
        get_property.vector(&optical.properties.refractive_index,
                            "RINDEX",
                            ImportUnits::unitless);

        // Save scintillation properties
        get_property.scalar(&optical.scintillation.material.yield_per_energy,
                            "SCINTILLATIONYIELD",
                            ImportUnits::inv_mev);
        get_property.scalar(&optical.scintillation.resolution_scale,
                            "RESOLUTIONSCALE",
                            ImportUnits::unitless);
        optical.scintillation.material.components
            = fill_vec_import_scint_comp(get_property);

        // Particle scintillation properties
        for (auto const& iter : particle_map)
        {
            auto const& particle_name = iter.first;

            ImportScintData::IPSS scint_part_spec;
            get_property.vector(&scint_part_spec.yield_vector,
                                particle_name + "SCINTILLATIONYIELD",
                                ImportUnits::inv_mev);
            scint_part_spec.components
                = fill_vec_import_scint_comp(get_property, particle_name);

            if (scint_part_spec)
            {
                optical.scintillation.particles.insert(
                    {iter.second.get(), std::move(scint_part_spec)});
            }
        }

        // Save Rayleigh properties
        get_property.vector(
            &optical.rayleigh.mfp, "RAYLEIGH", ImportUnits::len);
        get_property.scalar(&optical.rayleigh.scale_factor,
                            "RS_SCALE_FACTOR",
                            ImportUnits::unitless);
        get_property.scalar(&optical.rayleigh.compressibility,
                            "ISOTHERMAL_COMPRESSIBILITY",
                            ImportUnits::len_time_sq_per_mass);

        // Save absorption properties
        get_property.vector(&optical.absorption.absorption_length,
                            "ABSLENGTH",
                            ImportUnits::len);

        // Save WLS properties
        get_property.scalar(&optical.wls.mean_num_photons,
                            "WLSMEANNUMBERPHOTONS",
                            ImportUnits::unitless);
        get_property.scalar(
            &optical.wls.time_constant, "WLSTIMECONSTANT", ImportUnits::time);
        get_property.vector(
            &optical.wls.absorption_length, "WLSABSLENGTH", ImportUnits::len);
        get_property.vector(
            &optical.wls.component, "WLSCOMPONENT", ImportUnits::unitless);

        if (optical)
        {
            result.insert({mat_idx, std::move(optical)});
        }
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
import_phys_materials(GeantImporter::DataSelection::Flags particle_flags)
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
        // Fetch material, element, and production cuts lists
        auto const* mcc = pct.GetMaterialCutsCouple(i);
        CELER_ASSERT(mcc);
        CELER_ASSERT(static_cast<std::size_t>(mcc->GetIndex()) == i);

        auto const* g4material = mcc->GetMaterial();
        CELER_ASSERT(g4material);
        auto const* g4prod_cuts = mcc->GetProductionCuts();
        CELER_ASSERT(g4prod_cuts);

        // Populate material production cut values
        ImportPhysMaterial material;
        material.geo_material_id = g4material->GetIndex();
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
 * Return a populated \c ImportRegion vector.
 */
std::vector<ImportRegion> import_regions()
{
    auto& regions = *G4RegionStore::GetInstance();

    std::vector<ImportRegion> result(regions.size());

    // Loop over region data
    for (auto i : range(result.size()))
    {
        // Fetch material, element, and production cuts lists
        auto const* g4reg = regions[i];
        CELER_ASSERT(g4reg);
        CELER_ASSERT(static_cast<std::size_t>(g4reg->GetInstanceID()) == i);

        ImportRegion region;
        region.name = g4reg->GetName();
        region.field_manager = (g4reg->GetFieldManager() != nullptr);
        region.production_cuts = (g4reg->GetProductionCuts() != nullptr);
        region.user_limits = (g4reg->GetUserLimits() != nullptr);

        // Add region to result
        result[i] = std::move(region);
    }

    CELER_LOG(debug) << "Loaded " << result.size() << " regions";
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportProcess vector.
 */
auto import_processes(GeantImporter::DataSelection::Flags process_flags,
                      std::vector<ImportParticle> const& particles,
                      std::vector<ImportElement> const& elements,
                      std::vector<ImportPhysMaterial> const& materials)
    -> std::pair<std::vector<ImportProcess>, std::vector<ImportMscModel>>
{
    ParticleFilter include_particle{process_flags};
    ProcessFilter include_process{process_flags};

    std::vector<ImportProcess> processes;
    std::vector<ImportMscModel> msc_models;

    static celeritas::TypeDemangler<G4VProcess> const demangle_process;
    std::unordered_map<G4VProcess const*, G4ParticleDefinition const*> visited;
    detail::GeantProcessImporter import_process(
        detail::TableSelection::minimal, materials, elements);

    auto append_process = [&](G4ParticleDefinition const& particle,
                              G4VProcess const& process) -> void {
        // Check for duplicate processes
        auto [prev, inserted] = visited.insert({&process, &particle});

        if (!inserted)
        {
            CELER_LOG(debug)
                << "Skipping process '" << process.GetProcessName()
                << "' (RTTI: " << demangle_process(process)
                << ") for particle " << particle.GetParticleName()
                << ": duplicate of particle "
                << prev->second->GetParticleName();
            return;
        }

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
                    processes.push_back(import_process(particle, *subprocess));
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
            processes.push_back(import_process(particle, *em_process));
        }
        else if (auto const* el_process
                 = dynamic_cast<G4VEnergyLossProcess const*>(&process))
        {
            processes.push_back(import_process(particle, *el_process));
        }
        else if (auto const* msc_process
                 = dynamic_cast<G4VMultipleScattering const*>(&process))
        {
            // Unpack MSC process into multiple MSC models
            auto new_msc_models = import_process(particle, *msc_process);
            msc_models.insert(msc_models.end(),
                              std::make_move_iterator(new_msc_models.begin()),
                              std::make_move_iterator(new_msc_models.end()));
        }
        else
        {
            CELER_LOG(error)
                << "Cannot export unknown process '"
                << process.GetProcessName()
                << "' (RTTI: " << demangle_process(process) << ")";
        }
    };

    for (auto const& p : particles)
    {
        G4ParticleDefinition const* g4_particle_def
            = G4ParticleTable::GetParticleTable()->FindParticle(p.pdg);
        CELER_ASSERT(g4_particle_def);

        if (!include_particle(PDGNumber{g4_particle_def->GetPDGEncoding()}))
        {
            CELER_LOG(debug) << "Filtered all processes from particle '"
                             << g4_particle_def->GetParticleName() << "'";
            continue;
        }

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

            append_process(*g4_particle_def, process);
        }
    }
    CELER_LOG(debug) << "Loaded " << processes.size() << " processes";
    return {std::move(processes), std::move(msc_models)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the transportation process for a given particle type.
 */
G4Transportation const* get_transportation(G4ParticleDefinition const* particle)
{
    CELER_EXPECT(particle);

    auto const* pm = particle->GetProcessManager();
    CELER_ASSERT(pm);

    // Search through the processes to find transportion (it should be the
    // first one)
    auto const& pl = *pm->GetProcessList();
    for (auto i : range(pl.size()))
    {
        if (auto const* trans = dynamic_cast<G4Transportation const*>(pl[i]))
        {
            return trans;
        }
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Store particle-dependent transportation parameters.
 */
ImportTransParameters
import_trans_parameters(GeantImporter::DataSelection::Flags particle_flags)
{
    ImportTransParameters result;

    // Get the maximum number of substeps in the field propagator
    auto const* tm = G4TransportationManager::GetTransportationManager();
    CELER_ASSERT(tm);
    if (auto const* fp = tm->GetPropagatorInField())
    {
        result.max_substeps = fp->GetMaxLoopCount();
    }

    G4ParticleTable::G4PTblDicIterator& particle_iterator
        = *(G4ParticleTable::GetParticleTable()->GetIterator());
    particle_iterator.reset();
    ParticleFilter include_particle{particle_flags};
    while (particle_iterator())
    {
        auto const* particle = particle_iterator.value();
        if (!include_particle(PDGNumber{particle->GetPDGEncoding()}))
        {
            continue;
        }

        // Get the transportation process
        auto const* trans = get_transportation(particle);
        CELER_ASSERT(trans);

        // Get the threshold values for killing looping tracks
        ImportLoopingThreshold looping;
        looping.threshold_trials = trans->GetThresholdTrials();
        looping.important_energy = trans->GetThresholdImportantEnergy()
                                   * mev_scale;
        CELER_ASSERT(looping);
        result.looping.insert({particle->GetPDGEncoding(), looping});
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
    import.auger = g4.Auger();
    import.msc_step_algorithm = to_msc_step_algorithm(g4.MscStepLimitType());
    import.msc_range_factor = g4.MscRangeFactor();
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
    CELER_VALIDATE(world,
                   << "no world volume has been defined in the navigator");
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
    CELER_VALIDATE(
        (selected.materials && selected.particles != DataSelection::none)
            || selected.processes == DataSelection::none,
        << "materials and particles must be enabled if requesting processes");
    ScopedMem record_mem("GeantImporter.load");
    ScopedProfiling profile_this{"import-geant"};
    ImportData imported;

    {
        CELER_LOG(status) << "Transferring data from Geant4";
        ScopedGeantExceptionHandler scoped_exceptions;
        ScopedTimeLog scoped_time;
        if (selected.particles != DataSelection::none)
        {
            imported.particles = import_particles(selected.particles);
        }
        if (selected.materials)
        {
            imported.isotopes = import_isotopes();
            imported.elements = import_elements();
            imported.geo_materials = import_geo_materials();
            imported.phys_materials = import_phys_materials(selected.particles);
            imported.optical = import_optical();
        }
        if (selected.processes != DataSelection::none)
        {
            std::tie(imported.processes, imported.msc_models)
                = import_processes(selected.processes,
                                   imported.particles,
                                   imported.elements,
                                   imported.phys_materials);
        }
        imported.regions = import_regions();
        imported.volumes = this->import_volumes(selected.unique_volumes);
        if (selected.particles != DataSelection::none)
        {
            imported.trans_params = import_trans_parameters(selected.particles);
        }
        if (selected.processes & DataSelection::em)
        {
            imported.em_params = import_em_parameters();
        }
    }

    if (selected.reader_data)
    {
        CELER_LOG(status) << "Loading external elemental data";
        ScopedTimeLog scoped_time;

        detail::AllElementReader load_data{imported.elements};

        auto have_process = [&imported](ImportProcessClass ipc) {
            return std::any_of(imported.processes.begin(),
                               imported.processes.end(),
                               [ipc](const ImportProcess& ip) {
                                   return ip.process_class == ipc;
                               });
        };

        if (have_process(ImportProcessClass::e_brems))
        {
            imported.sb_data = load_data(SeltzerBergerReader{});
        }
        if (have_process(ImportProcessClass::photoelectric))
        {
            imported.livermore_pe_data = load_data(LivermorePEReader{});
        }
        if (G4EmParameters::Instance()->Fluo())
        {
            // TODO: only read auger data if that option is enabled
            imported.atomic_relaxation_data
                = load_data(AtomicRelaxationReader{});
        }
        else if (G4EmParameters::Instance()->Auger())
        {
            CELER_LOG(warning) << "Auger emission is ignored because "
                                  "fluorescent atomic relaxation is disabled";
        }
    }

    imported.units = units::NativeTraits::label();
    return imported;
}

//---------------------------------------------------------------------------//
/*!
 * Return a populated \c ImportVolume vector.
 */
std::vector<ImportVolume>
GeantImporter::import_volumes(bool unique_volumes) const
{
    // Note: if the LV has been purged (i.e. by trying to run multiple
    // geometries in the same execution), the instance ID's won't correspond to
    // the location in the vector.
    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(lv_store);
    std::vector<ImportVolume> result;
    result.reserve(lv_store->size());

    // Recursive loop over all logical volumes to populate volumes
    int count = 0;
    visit_geant_volumes(
        [unique_volumes, &result, &count](G4LogicalVolume const& lv) {
            auto i = static_cast<std::size_t>(lv.GetInstanceID());
            if (i >= result.size())
            {
                result.resize(i + 1);
            }
            ++count;

            ImportVolume& volume = result[lv.GetInstanceID()];
            if (auto* mat = lv.GetMaterial())
            {
                volume.geo_material_id = mat->GetIndex();
            }
            if (auto* reg = lv.GetRegion())
            {
                volume.region_id = reg->GetInstanceID();
            }
            if (auto* cuts = lv.GetMaterialCutsCouple())
            {
                volume.phys_material_id = cuts->GetIndex();
            }
            volume.name = lv.GetName();
            volume.solid_name = lv.GetSolid()->GetName();

            if (volume.name.empty())
            {
                CELER_LOG(warning)
                    << "No logical volume name specified for instance ID " << i
                    << " (material " << volume.phys_material_id << ")";
            }
            else if (unique_volumes)
            {
                // Add pointer as GDML writer does
                volume.name = make_gdml_name(lv);
            }
        },
        *world_->GetLogicalVolume());

    CELER_LOG(debug) << "Loaded " << count << " volumes with "
                     << (unique_volumes ? "uniquified" : "original")
                     << " names";
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

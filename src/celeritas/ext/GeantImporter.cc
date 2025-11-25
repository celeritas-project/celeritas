//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include <G4Cerenkov.hh>
#include <G4Element.hh>
#include <G4ElementTable.hh>
#include <G4ElementVector.hh>
#include <G4EmParameters.hh>
#include <G4GammaGeneralProcess.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Material.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4MscStepLimitType.hh>
#include <G4MuPairProduction.hh>
#include <G4MuPairProductionModel.hh>
#include <G4Navigator.hh>
#include <G4NuclearFormfactorType.hh>
#include <G4NucleiProperties.hh>
#include <G4OpAbsorption.hh>
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
#include <G4TransportationManager.hh>
#include <G4Types.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4VMultipleScattering.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VProcess.hh>
#include <G4VRangeToEnergyConverter.hh>
#include <G4Version.hh>

#include "celeritas/io/ImportUnits.hh"
#if G4VERSION_NUMBER >= 1070
#    include <G4OpWLS2.hh>
#    include <G4OpticalParameters.hh>
#endif

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/math/PdfUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "GeantParticleView.hh"
#include "GeantSetup.hh"

#include "detail/GeantMaterialPropertyGetter.hh"
#include "detail/GeantOpticalModelImporter.hh"
#include "detail/GeantProcessImporter.hh"
#include "detail/GeantSurfacePhysicsLoader.hh"

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

    DataSelection::Flags which;

    bool operator()(G4ProcessType pt)
    {
        switch (pt)
        {
            case G4ProcessType::fElectromagnetic:
                return (which & DataSelection::em);
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
//! Custom-defined scintillation properties approximating to Gaussian
//! distribution
ImportGaussianScintComponent
load_gauss_scint(std::string const& prefix,
                 detail::GeantMaterialPropertyGetter& get_property,
                 int comp_idx)
{
    ImportGaussianScintComponent gauss;
    bool found_mean = false;

    for (auto&& [prop, label] : {std::pair{&gauss.lambda_mean, "LAMBDAMEAN"},
                                 std::pair{&gauss.lambda_sigma, "LAMBDASIGMA"}})
    {
        if (get_property(
                prop, "CELER_" + prefix + label, comp_idx, ImportUnits::len))
        {
            found_mean = true;
        }
    }
    if (!found_mean)
    {
        return {};
    }
    return gauss;
}

//---------------------------------------------------------------------------//
/*!
 * Populate an \c ImportScintComponent .
 * To retrieve a material-only component simply do not use particle name.
 */
std::vector<ImportScintComponent>
fill_vec_import_scint_comp(detail::GeantMaterialPropertyGetter& get_property,
                           std::string prefix = {})
{
    CELER_EXPECT(prefix.empty() || optical_particles_map().count(prefix));

    // All the components below are "SCINTILLATIONYIELD",
    // "ELECTRONSCINTILLATIONYIELD", etc.
    prefix += "SCINTILLATION";

    std::vector<ImportScintComponent> components;
    for (int comp_idx : range(1, 4))
    {
        bool any_found = false;
        auto get = [&](double* dst, std::string const& ext, ImportUnits u) {
            bool one_found = get_property(dst, prefix + ext, comp_idx, u);
            any_found = any_found || one_found;
            return one_found;
        };

        ImportScintComponent comp;
        get(&comp.yield_frac, "YIELD", ImportUnits::inv_mev);

        // Rise time is not defined for particle type in Geant4
        get(&comp.rise_time, "RISETIME", ImportUnits::time);
        get(&comp.fall_time, "TIMECONSTANT", ImportUnits::time);

        auto name = prefix + "COMPONENT" + std::to_string(comp_idx);
        inp::Grid grid;

        using ScintSpectrumComponent
            = std::variant<std::monostate, ImportGaussianScintComponent, inp::Grid>;
        ScintSpectrumComponent spectrum_component;

        if (auto gauss
            = load_gauss_scint("CELER_" + prefix, get_property, comp_idx))
        {
            CELER_VALIDATE(
                std::holds_alternative<std::monostate>(spectrum_component),
                << "conflicting scintillation spectrum definitions "
                   "for "
                << prefix);
            spectrum_component = std::move(gauss);
        }
        if (auto gauss = load_gauss_scint(prefix, get_property, comp_idx))
        {
            CELER_VALIDATE(
                std::holds_alternative<std::monostate>(spectrum_component),
                << "conflicting/redundant scintillation properties "
                   "for "
                << prefix);
            CELER_LOG(warning) << "Deprecated property prefix " << prefix
                               << "LAMBDA: use CELER_" << prefix;
            spectrum_component = std::move(gauss);
        }
        if (get_property(&grid, name, {ImportUnits::mev, ImportUnits::unitless}))
        {
            // If an explicit energy/intensity grid is provided, use it
            spectrum_component = std::move(grid);
        }

        if (auto* g
            = std::get_if<ImportGaussianScintComponent>(&spectrum_component))
        {
            comp.gauss = *g;
        }
        else if (auto* gr = std::get_if<inp::Grid>(&spectrum_component))
        {
            comp.spectrum = *gr;
        }

        bool has_spectrum
            = !std::holds_alternative<std::monostate>(spectrum_component);
        if (any_found || has_spectrum)
            // Note that the user may be missing some properties: in that
            // case (if Geant4 didn't warn/error/die already) then we will
            // rely on the downstream code to validate.
            // Additionally, this check prevents adding components with only
            // default (zero) values and no spectrum, which would otherwise
            // trigger validation errors.
            components.push_back(std::move(comp));
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

        if (!include_particle(particle_view.pdg()))
        {
            continue;
        }

        particles.push_back([&particle_view] {
            inp::Particle result;
            result.name = particle_view.name();
            result.pdg = particle_view.pdg();
            result.mass = particle_view.mass();
            result.charge = particle_view.charge();
            result.decay_constant = particle_view.decay_constant();

            if (G4VERSION_NUMBER < 1070 && particle_view.is_optical_photon())
            {
                // Before 10.7, geant4 uses PDG 0 plus a unique string name
                result.pdg = g4_optical_pdg;
            }

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
import_optical_materials(detail::GeoOpticalIdMap const& geo_to_opt)
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
        auto const* mpt = material->GetMaterialPropertiesTable();
        CELER_ASSERT(mpt);

        detail::GeantMaterialPropertyGetter get_property{*mpt};

        // Optical materials should map uniquely
        ImportOpticalMaterial& optical = result[opt_mat_id.get()];
        CELER_ASSERT(!optical);

        // Save common properties
        bool has_rindex
            = get_property(&optical.properties.refractive_index,
                           "RINDEX",
                           {ImportUnits::mev, ImportUnits::unitless});
        // Existence of RINDEX should correspond to GeoOpticalIdMap
        // construction
        CELER_ASSERT(has_rindex);

        // Save scintillation properties
        get_property(&optical.scintillation.material.yield_per_energy,
                     "SCINTILLATIONYIELD",
                     ImportUnits::inv_mev);
        get_property(&optical.scintillation.resolution_scale,
                     "RESOLUTIONSCALE",
                     ImportUnits::unitless);
        optical.scintillation.material.components
            = fill_vec_import_scint_comp(get_property);

        // Particle scintillation properties
        for (auto&& [prefix, pdg] : optical_particles_map())
        {
            ImportScintData::IPSS scint_part_spec;
            get_property(&scint_part_spec.yield_vector,
                         prefix + "SCINTILLATIONYIELD",
                         {ImportUnits::mev, ImportUnits::inv_mev});
            scint_part_spec.components
                = fill_vec_import_scint_comp(get_property, prefix);

            if (scint_part_spec)
            {
                optical.scintillation.particles.insert(
                    {pdg.get(), std::move(scint_part_spec)});
            }
        }

        // Save Rayleigh properties
        get_property(&optical.rayleigh.scale_factor,
                     "RS_SCALE_FACTOR",
                     ImportUnits::unitless);
        get_property(&optical.rayleigh.compressibility,
                     "ISOTHERMAL_COMPRESSIBILITY",
                     ImportUnits::len_time_sq_per_mass);
        if (optical.rayleigh.compressibility == 0
            && material->GetName() == "Water")
        {
            // Use special default hardcoded value for water
            // in G4OpRayleigh::CalculateRayleighMeanFreePaths
            using CLHEP::m3;
            using CLHEP::MeV;
            double const betat = 7.658e-23 * m3 / MeV;
            optical.rayleigh.compressibility
                = betat
                  * native_value_from_clhep(ImportUnits::len_time_sq_per_mass);
            CELER_LOG(info) << "Setting compressibility of water to "
                            << optical.rayleigh.compressibility << " m^2/N";

            if (!soft_equal(material->GetTemperature(), 283.15 * CLHEP::kelvin))
            {
                CELER_LOG(warning)
                    << "Geant4 Rayleigh optical scattering ignores material "
                       "temperature for Water (overriding "
                    << material->GetTemperature()
                    << " K with 283.15 K) if no `RAYLEIGH` mean free paths "
                       "are provided";
            }
        }

        // Save WLS properties
        get_property(&optical.wls.mean_num_photons,
                     "WLSMEANNUMBERPHOTONS",
                     ImportUnits::unitless);
        get_property(
            &optical.wls.time_constant, "WLSTIMECONSTANT", ImportUnits::time);
        get_property(&optical.wls.component,
                     "WLSCOMPONENT",
                     {ImportUnits::mev, ImportUnits::unitless});

        // Save WLS2 properties
        get_property(&optical.wls2.mean_num_photons,
                     "WLSMEANNUMBERPHOTONS2",
                     ImportUnits::unitless);
        get_property(
            &optical.wls2.time_constant, "WLSTIMECONSTANT2", ImportUnits::time);
        get_property(&optical.wls2.component,
                     "WLSCOMPONENT2",
                     {ImportUnits::mev, ImportUnits::unitless});

        // Save Mie properties
        get_property(&optical.mie.forward_ratio,
                     "MIEHG_FORWARD_RATIO",
                     ImportUnits::unitless);
        get_property(
            &optical.mie.forward_g, "MIEHG_FORWARD", ImportUnits::unitless);
        get_property(
            &optical.mie.backward_g, "MIEHG_BACKWARD", ImportUnits::unitless);

        CELER_ASSERT(optical);
    }

    CELER_LOG(debug) << "Loaded " << result.size() << " optical materials";
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Import optical surface physics information.
 */
inp::SurfacePhysics import_optical_surface_physics()
{
    inp::SurfacePhysics result;
    auto geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geo, << "global Geant4 geometry is not loaded");

    MultiExceptionHandler handle;
    detail::GeantSurfacePhysicsLoader load_surface(result);
    for (auto sid : range(SurfaceId(geo->num_surfaces())))
    {
        CELER_TRY_HANDLE(load_surface(sid), handle);
    }
    log_and_rethrow(std::move(handle));

    // Default Geant4 surface
    size_type num_phys_surfaces{0};
    for (auto const& mats : result.materials)
    {
        num_phys_surfaces += mats.size() + 1;
    }
    PhysSurfaceId default_surface(num_phys_surfaces);
    result.materials.push_back({});
    result.roughness.polished.emplace(default_surface, inp::NoRoughness{});
    result.reflectivity.fresnel.emplace(default_surface,
                                        inp::FresnelReflection{});
    result.interaction.dielectric.emplace(
        default_surface,
        inp::DielectricInteraction::from_dielectric(
            inp::ReflectionForm::from_spike()));

    CELER_LOG(debug) << "Loaded " << result.materials.size()
                     << " optical surfaces (" << num_phys_surfaces
                     << " physics surfaces)";
    CELER_ENSURE(result || (geo->num_surfaces() == 0));
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
                      detail::GeoOpticalIdMap const& geo_to_opt)
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
 * Return a populated \c ImportProcess vector.
 */
auto import_processes(GeantImporter::DataSelection selected,
                      detail::GeoOpticalIdMap const& geo_to_opt,
                      ImportData& imported)
{
    ParticleFilter include_particle{selected.processes};
    ProcessFilter include_process{selected.processes};

    auto const& particles = imported.particles;
    auto const& elements = imported.elements;
    auto const& materials = imported.phys_materials;

    auto& processes = imported.processes;
    auto& msc_models = imported.msc_models;
    auto& optical_models = imported.optical_models;

    static celeritas::TypeDemangler<G4VProcess> const demangle_process;
    std::unordered_map<G4VProcess const*, G4ParticleDefinition const*> visited;
    detail::GeantProcessImporter import_process(
        materials, elements, selected.interpolation);
    detail::GeantOpticalModelImporter import_optical_model(geo_to_opt);

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

        // TODO: change this to a map of processes like g4vg/g4org

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
        else if (import_optical_model
                 && dynamic_cast<G4OpAbsorption const*>(&process))
        {
            optical_models.push_back(
                import_optical_model(optical::ImportModelClass::absorption));
        }
        else if (import_optical_model
                 && dynamic_cast<G4OpRayleigh const*>(&process))
        {
            optical_models.push_back(
                import_optical_model(optical::ImportModelClass::rayleigh));
        }
        else if (import_optical_model && dynamic_cast<G4OpWLS const*>(&process))
        {
            optical_models.push_back(
                import_optical_model(optical::ImportModelClass::wls));
        }
        else if (import_optical_model
                 && dynamic_cast<G4OpMieHG const*>(&process))
        {
            optical_models.push_back(
                import_optical_model(optical::ImportModelClass::mie));
        }

#if G4VERSION_NUMBER >= 1070
        else if (import_optical_model
                 && dynamic_cast<G4OpWLS2 const*>(&process))
        {
            optical_models.push_back(
                import_optical_model(optical::ImportModelClass::wls2));
        }
#endif
        else if (dynamic_cast<G4Cerenkov const*>(&process))
        {
            imported.optical_physics.cherenkov = true;
        }
        else if (dynamic_cast<G4Scintillation const*>(&process))
        {
            imported.optical_physics.scintillation = true;
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
            = G4ParticleTable::GetParticleTable()->FindParticle(p.pdg.get());
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
                continue;
            }

            append_process(*g4_particle_def, process);
        }
    }

    // Optical photon PDG in Geant4 is 0 before version 10.7
    if (G4VERSION_NUMBER < 1070
        && G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton"))
    {
        auto* photon_def = G4OpticalPhoton::OpticalPhoton();
        CELER_ASSERT(photon_def);

        if (!include_particle(g4_optical_pdg))
        {
            CELER_LOG(debug) << "Filtered all processes from particle '"
                             << photon_def->GetParticleName() << "'";
        }
        else
        {
            G4ProcessVector const& process_list
                = *photon_def->GetProcessManager()->GetProcessList();

            for (auto j : range(process_list.size()))
            {
                G4VProcess const& process = *process_list[j];
                if (!include_process(process.GetProcessType()))
                {
                    continue;
                }

                append_process(*photon_def, process);
            }
        }
    }

    CELER_LOG(debug) << "Loaded " << processes.size() << " processes";
    CELER_LOG(debug) << "Loaded " << msc_models.size() << " msc models";
    CELER_LOG(debug) << "Loaded " << optical_models.size() << " optical models";
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
 * Import optical parameters.
 */
ImportOpticalParameters import_optical_parameters()
{
    ImportOpticalParameters iop;

#if G4VERSION_NUMBER >= 1070
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);

    auto to_enum = [](std::string const& time_profile) {
        if (time_profile == "delta")
        {
            return WlsTimeProfile::delta;
        }
        else if (time_profile == "exponential")
        {
            return WlsTimeProfile::exponential;
        }
        CELER_NOT_IMPLEMENTED("unknown WLS time profile " + time_profile);
    };
    iop.wls_time_profile = to_enum(params->GetWLSTimeProfile());
    iop.wls2_time_profile = to_enum(params->GetWLS2TimeProfile());

    //! \todo Set \c scintillation_by_particle when supported
    //! \todo For older Geant4 versions, set based on user input?
#endif

    return iop;
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
 * Get the sampling table for electron-positron pair production by muons.
 */
inp::MuPairProductionEnergyTransferTable import_mupp_table(PDGNumber pdg)
{
    CELER_EXPECT(pdg == pdg::mu_minus() || pdg == pdg::mu_plus());

    using IU = ImportUnits;

    G4ParticleDefinition const* pdef
        = G4ParticleTable::GetParticleTable()->FindParticle(pdg.get());
    CELER_ASSERT(pdef);

    auto const* process = dynamic_cast<G4MuPairProduction const*>(
        pdef->GetProcessManager()->GetProcess(
            to_geant_name(ImportProcessClass::mu_pair_prod)));
    CELER_ASSERT(process);
    CELER_ASSERT(process->NumberOfModels() == 1);

    auto* model = dynamic_cast<G4MuPairProductionModel*>(process->EmModel());
    CELER_ASSERT(model);

    G4ElementData* el_data = model->GetElementData();
    CELER_ASSERT(el_data);

    inp::MuPairProductionEnergyTransferTable result;
    if (G4VERSION_NUMBER < 1120)
    {
        constexpr int element_data_size = 99;
        for (int z = 1; z < element_data_size; ++z)
        {
            if (G4Physics2DVector const* pv = el_data->GetElement2DData(z))
            {
                result.atomic_number.push_back(AtomicNumber{z});
                result.grids.push_back(detail::import_physics_2dvector(
                    *pv, {IU::unitless, IU::mev, IU::mev_len_sq}));
            }
        }
    }
    else
    {
        // The muon pair production model in newer Geant4 versions initializes
        // and accesses the element data by Z index rather than Z number
        using Z = AtomicNumber;
        result.atomic_number = {Z{1}, Z{4}, Z{13}, Z{29}, Z{92}};
        for (int i : range(result.atomic_number.size()))
        {
            G4Physics2DVector const* pv = el_data->GetElement2DData(i);
            CELER_ASSERT(pv);
            result.grids.push_back(detail::import_physics_2dvector(
                *pv, {IU::unitless, IU::mev, IU::mev_len_sq}));
        }
    }

    CELER_ENSURE(result);
    return result;
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

    auto have_process = [&imported](ImportProcessClass ipc) {
        return std::any_of(imported.processes.begin(),
                           imported.processes.end(),
                           [ipc](ImportProcess const& ip) {
                               return ip.process_class == ipc;
                           });
    };

    {
        CELER_LOG(status) << "Transferring data from Geant4";
        ScopedGeantExceptionHandler scoped_exceptions;
        ScopedTimeLog scoped_time;

        detail::GeoOpticalIdMap geo_to_opt;

        if (selected.particles != DataSelection::none)
        {
            imported.particles = import_particles(selected.particles);
        }
        if (selected.materials)
        {
            if (selected.processes & DataSelection::optical)
            {
                geo_to_opt
                    = detail::GeoOpticalIdMap(*G4Material::GetMaterialTable());
                imported.optical_materials
                    = import_optical_materials(geo_to_opt);
            }

            imported.isotopes = import_isotopes();
            imported.elements = import_elements();
            imported.geo_materials = import_geo_materials();
            imported.phys_materials
                = import_phys_materials(selected.particles, geo_to_opt);
        }
        if (selected.processes != DataSelection::none)
        {
            import_processes(selected, geo_to_opt, imported);

            if (have_process(ImportProcessClass::mu_pair_prod))
            {
                auto mu_minus = import_mupp_table(pdg::mu_minus());
                auto mu_plus = import_mupp_table(pdg::mu_plus());
                CELER_VALIDATE(mu_minus.atomic_number == mu_plus.atomic_number
                                   && mu_minus.grids == mu_plus.grids,
                               << "muon pair production sampling tables for "
                                  "mu- and mu+ differ");
                inp::MuPairProductionModel mupp_model;
                mupp_model.muppet_table = std::move(mu_minus);
                imported.mu_production = std::move(mupp_model);
            }
        }
        if (selected.unique_volumes)
        {
            // TODO: remove in v0.7
            CELER_LOG(warning)
                << R"(DEPRECATED: volumes are always reproducibly uniquified)";
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
        if (selected.processes & DataSelection::optical)
        {
            imported.optical_params = import_optical_parameters();
            imported.optical_physics.surfaces
                = import_optical_surface_physics();
        }
    }

    imported.units = units::NativeTraits::label();
    return imported;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

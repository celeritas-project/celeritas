//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsLoader.hh"

#include <typeindex>
#include <unordered_map>
#include <G4Cerenkov.hh>
#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4MuonMinusAtomicCapture.hh>
#include <G4OpAbsorption.hh>
#include <G4OpBoundaryProcess.hh>
#include <G4OpMieHG.hh>
#include <G4OpRayleigh.hh>
#include <G4OpWLS.hh>
#include <G4Scintillation.hh>
#include <G4VProcess.hh>
#include <G4Version.hh>

#if G4VERSION_NUMBER >= 1070
#    include <G4OpWLS2.hh>
#    include <G4OpticalParameters.hh>
#endif

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/Types.hh"

#include "GeantMaterialPropertyGetter.hh"
#include "GeantSurfacePhysicsLoader.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
#if G4VERSION_NUMBER >= 1070
//! Convert a G4 WLS string to a distribution enum
optical::WlsDistribution geant_to_wls_distribution(std::string const& s)
{
    static auto const from_string
        = StringEnumMapper<optical::WlsDistribution>::from_cstring_func(
            optical::to_cstring, "wls distribution");

    return from_string(s);
}
#else
// WLS2 is not available: define a dummy process in the anonymous namespace to
// allow us to build the dipatch table below
class G4OpWLS2 : public G4OpWLS
{
};
#endif

//---------------------------------------------------------------------------//
/*!
 * Set special default hardcoded value for water.
 *
 * This is from an implementation detail in \c
 * G4OpRayleigh::CalculateRayleighMeanFreePaths .
 */
void load_rayleigh_water(
    inp::OpticalModelMaterial<ImportOpticalRayleigh>& model_mat,
    G4Material const& g4mat)
{
    double const betat = 7.658e-23 * CLHEP::m3 / CLHEP::MeV;
    constexpr auto units = ImportUnits::len_time_sq_per_mass;
    model_mat.compressibility = betat * native_value_from_clhep(units);
    CELER_LOG(warning) << "DEPRECATED: using Geant4 built-in Rayleigh "
                          "properties for water: setting to "
                       << model_mat.compressibility << " "
                       << to_cstring(units);

    if (!soft_equal(g4mat.GetTemperature(), 283.15 * CLHEP::kelvin))
    {
        CELER_LOG(error)
            << "Geant4 Rayleigh optical scattering ignores material "
               "temperature for Water (overriding "
            << g4mat.GetTemperature()
            << " K with 283.15 K) if no `RAYLEIGH` mean free paths "
               "are provided";
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with reference to data being loaded.
 */
GeantPhysicsLoader::GeantPhysicsLoader(ImportData& imported,
                                       GeoOpticalIdMap const& optical_ids)
    : imported_{imported}, optical_ids_{optical_ids}
{
    if (!optical_ids.empty())
    {
        // Most optical physics data is stored as a material property and is
        // material-dependent
        optical_g4mat_.assign(optical_ids.num_optical(), nullptr);
        auto const& mt = *G4Material::GetMaterialTable();
        for (auto geo_mat_id : range(id_cast<GeoMatId>(mt.size())))
        {
            auto opt_id = optical_ids[geo_mat_id];
            if (!opt_id)
            {
                continue;
            }
            G4Material const* material = mt[geo_mat_id.get()];
            CELER_ASSERT(material);
            optical_g4mat_[opt_id.get()] = material;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Load data from a process, returning whether it was recognized.
 *
 * Returns \c true if the process type is known (whether or not data was
 * imported), and \c false if it was not recognized.
 */
bool GeantPhysicsLoader::operator()(G4VProcess const& p)
{
    if (!visited_.insert(&p).second)
    {
        // Already inserted
        CELER_LOG(debug) << "Already loaded process \"" << p.GetProcessName()
                         << "\"";
        return true;
    }

    using MemberFuncPtr = size_type (GeantPhysicsLoader::*)(G4VProcess const&);
    using PairNameMfptr = std::pair<char const*, MemberFuncPtr>;
    using TypeHandlerMap = std::unordered_map<std::type_index, PairNameMfptr>;

    // clang-format off
#define GPL_TYPE_FUNC(CLASSNAME, METHOD) \
    {std::type_index(typeid(CLASSNAME)), {#CLASSNAME, &GeantPhysicsLoader::METHOD}}
    static TypeHandlerMap const type_to_handler{
        // EM particles
        GPL_TYPE_FUNC(G4Cerenkov,               cerenkov),
        GPL_TYPE_FUNC(G4MuonMinusAtomicCapture, muon_minus_atomic_capture),
        GPL_TYPE_FUNC(G4Scintillation,          scintillation),
        // Optical photons
        GPL_TYPE_FUNC(G4OpAbsorption,      op_absorption),
        GPL_TYPE_FUNC(G4OpBoundaryProcess, op_boundary),
        GPL_TYPE_FUNC(G4OpMieHG,           op_mie_hg),
        GPL_TYPE_FUNC(G4OpRayleigh,        op_rayleigh),
        GPL_TYPE_FUNC(G4OpWLS,             op_wls),
        GPL_TYPE_FUNC(G4OpWLS2,            op_wls2),
    };
    // clang-format on
#undef GPL_TYPE_FUNC

    auto iter = type_to_handler.find(std::type_index(typeid(p)));
    if (iter == type_to_handler.end())
    {
        // Unknown process: let someone else handle it
        return false;
    }
    auto&& [name, mfptr] = iter->second;
    size_type result{0};
    try
    {
        result = (this->*mfptr)(p);
    }
    catch (...)
    {
        CELER_LOG(error) << "Failed while loading process " << name << "(\""
                         << p.GetProcessName() << "\")";
        throw;
    }

    auto msg
        = world_logger()(CELER_CODE_PROVENANCE,
                         result == 0 ? LogLevel::warning : LogLevel::debug);
    msg << "Loaded ";
    if (result == 0)
    {
        msg << "no";
    }
    else
    {
        msg << result;
    }
    msg << " model data from process " << name << "(\"" << p.GetProcessName()
        << "\")";
    return true;
}

//---------------------------------------------------------------------------//
//! Load Cherenkov emission (TODO: enable by material)
size_type GeantPhysicsLoader::cerenkov(G4VProcess const&)
{
    auto& model = imported_.optical_physics.cherenkov;
    model = true;
    return 1;
}

//---------------------------------------------------------------------------//
//! Initialize muon-catalyzed fusion
size_type GeantPhysicsLoader::muon_minus_atomic_capture(G4VProcess const&)
{
    auto& model = imported_.mucf_physics;
    // G4MuonMinusAtomicCapture is a G4ProcessType::fHadronic
    // It is also a G4VRestProcess and does not require import data
    model = inp::MucfPhysics::from_default();
    return model.cycle_rates.size();
}

//---------------------------------------------------------------------------//
//! Load optical scintillation
size_type GeantPhysicsLoader::scintillation(G4VProcess const&)
{
    auto& model = imported_.optical_physics.scintillation;
    model = true;
    // TODO: load materials/spectra
    return 1;
}

//---------------------------------------------------------------------------//
//! Load optical absorption
size_type GeantPhysicsLoader::op_absorption(G4VProcess const&)
{
    auto& model = imported_.optical_physics.bulk.absorption;
    this->load_mfps(model, "ABSLENGTH");
    return model.materials.size();
}

//---------------------------------------------------------------------------//
//! Load optical surface physics
size_type GeantPhysicsLoader::op_boundary(G4VProcess const&)
{
    auto& materials = imported_.optical_materials;
    if (materials.empty())
    {
        CELER_LOG(error) << "Optical boundary process is defined but no "
                            "optical materials are present";
        return 0;
    }

    auto& surfaces = imported_.optical_physics.surfaces;

    // Load each geometry surface and print any errors that occur
    {
        auto geo = celeritas::global_geant_geo().lock();
        CELER_VALIDATE(geo, << "global Geant4 geometry is not loaded");

        MultiExceptionHandler handle;
        detail::GeantSurfacePhysicsLoader load_surface(surfaces, materials);
        for (auto sid : range(SurfaceId(geo->num_surfaces())))
        {
            CELER_TRY_HANDLE(load_surface(sid), handle);
        }
        log_and_rethrow(std::move(handle));
    }

    // Add default Geant4 surface
    size_type num_phys_surfaces{0};
    for (auto const& mats : surfaces.materials)
    {
        num_phys_surfaces += mats.size() + 1;
    }
    PhysSurfaceId default_surface(num_phys_surfaces);
    surfaces.materials.push_back({});
    surfaces.roughness.polished.emplace(default_surface, inp::NoRoughness{});
    surfaces.reflectivity.fresnel.emplace(default_surface,
                                          inp::FresnelReflection{});
    surfaces.interaction.dielectric.emplace(
        default_surface,
        inp::DielectricInteraction::from_dielectric(
            inp::ReflectionForm::from_spike()));

    CELER_ENSURE(surfaces);
    return surfaces.materials.size();
}

//---------------------------------------------------------------------------//
//! Load Mie scattering
size_type GeantPhysicsLoader::op_mie_hg(G4VProcess const&)
{
    auto& model = imported_.optical_physics.bulk.mie;
    this->load_mfps(model, "MIEHG");
    for (auto&& [opt_mat_id, model_mat] : model.materials)
    {
        auto get_property = this->property_getter(opt_mat_id);
        get_property(model_mat.forward_ratio,
                     "MIEHG_FORWARD_RATIO",
                     ImportUnits::unitless);
        get_property(
            model_mat.forward_g, "MIEHG_FORWARD", ImportUnits::unitless);
        get_property(
            model_mat.backward_g, "MIEHG_BACKWARD", ImportUnits::unitless);
    }
    return model.materials.size();
}

//---------------------------------------------------------------------------//
//! Load rayleigh scattering
size_type GeantPhysicsLoader::op_rayleigh(G4VProcess const&)
{
    // TODO: refactor as variant: MFP grid *or* scale_factor+compressibility
    auto& model = imported_.optical_physics.bulk.rayleigh;
    this->load_mfps(model, "RAYLEIGH");

    // Look for additional material data or special cases if MFP isn't provided
    // as a grid
    for (auto opt_mat_id : range(OptMatId{optical_ids_.num_optical()}))
    {
        if (model.materials.count(opt_mat_id))
        {
            continue;
        }

        auto get_property = this->property_getter(opt_mat_id);
        inp::OpticalModelMaterial<ImportOpticalRayleigh> model_mat;
        bool any_found = false;
        any_found = get_property(model_mat.scale_factor,
                                 "RS_SCALE_FACTOR",
                                 ImportUnits::unitless)
                    || any_found;
        any_found = get_property(model_mat.compressibility,
                                 "ISOTHERMAL_COMPRESSIBILITY",
                                 ImportUnits::len_time_sq_per_mass)
                    || any_found;
        if (!any_found)
        {
            auto& g4mat = *optical_g4mat_[opt_mat_id.get()];
            if (g4mat.GetName() == "Water")
            {
                load_rayleigh_water(model_mat, g4mat);
                any_found = true;
            }
        }
        if (any_found)
        {
            CELER_VALIDATE(model_mat, << "inconsistent Rayleigh input data");
            // Add non-grid rayleigh
            model.materials.emplace(opt_mat_id, std::move(model_mat));
        }
    }
    return model.materials.size();
}

//---------------------------------------------------------------------------//
//! Load wavelength shifting
size_type GeantPhysicsLoader::op_wls(G4VProcess const&)
{
#if G4VERSION_NUMBER >= 1070
    // Save time profile
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    imported_.optical_params.wls_time_profile
        = geant_to_wls_distribution(params->GetWLSTimeProfile());
#endif

    auto& model = imported_.optical_physics.bulk.wls;
    this->load_mfps(model, "WLSABSLENGTH");
    for (auto&& [opt_mat_id, model_mat] : model.materials)
    {
        auto get_property = this->property_getter(opt_mat_id);
        get_property(model_mat.mean_num_photons,
                     "WLSMEANNUMBERPHOTONS",
                     ImportUnits::unitless);
        get_property(
            model_mat.time_constant, "WLSTIMECONSTANT", ImportUnits::time);
        get_property(model_mat.component,
                     "WLSCOMPONENT",
                     {ImportUnits::mev, ImportUnits::unitless});
    }
    return model.materials.size();
}

//---------------------------------------------------------------------------//
//! Load wavelength shifting additional distribution
size_type GeantPhysicsLoader::op_wls2(G4VProcess const&)
{
#if G4VERSION_NUMBER >= 1070
    // Save time profile
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    imported_.optical_params.wls_time_profile
        = geant_to_wls_distribution(params->GetWLS2TimeProfile());

    auto& model = imported_.optical_physics.bulk.wls2;
    this->load_mfps(model, "WLSABSLENGTH2");
    for (auto&& [opt_mat_id, model_mat] : model.materials)
    {
        auto get_property = this->property_getter(opt_mat_id);
        get_property(model_mat.mean_num_photons,
                     "WLSMEANNUMBERPHOTONS2",
                     ImportUnits::unitless);
        get_property(
            model_mat.time_constant, "WLSTIMECONSTANT2", ImportUnits::time);
        get_property(model_mat.component,
                     "WLSCOMPONENT2",
                     {ImportUnits::mev, ImportUnits::unitless});
    }
    return model.materials.size();
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Access material properties for an optical material.
 */
auto GeantPhysicsLoader::property_getter(OptMatId opt_id) const -> PropGetter
{
    CELER_EXPECT(opt_id < optical_g4mat_.size());
    G4Material const* mat = optical_g4mat_[opt_id.get()];
    CELER_ASSERT(mat);
    return PropGetter{mat->GetMaterialPropertiesTable(), mat->GetName()};
}

//---------------------------------------------------------------------------//
/*!
 * Look up an MFP grid for the given optical material and property name.
 *
 * Returns an empty grid if no properties table exists for the material or
 * the property is not found.
 */
inp::Grid GeantPhysicsLoader::load_mfp(OptMatId opt_id,
                                       std::string const& prop_name) const
{
    auto get_property = this->property_getter(opt_id);
    inp::Grid mfp;
    get_property(mfp, prop_name, {ImportUnits::mev, ImportUnits::len});
    return mfp;
}

//---------------------------------------------------------------------------//
/*!
 * Loop over optical materials and populate model.materials with MFP grids.
 */
template<class MM, optical::ImportModelClass IMC>
void GeantPhysicsLoader::load_mfps(inp::OpticalBulkModel<MM, IMC>& model,
                                   std::string const& prop_name) const
{
    CELER_EXPECT(model.materials.empty());
    for (auto opt_id : range(OptMatId{optical_ids_.num_optical()}))
    {
        if (auto mfp = this->load_mfp(opt_id, prop_name))
        {
            inp::OpticalModelMaterial<MM> model_mat;
            model_mat.mfp = std::move(mfp);
            model.materials.emplace(opt_id, std::move(model_mat));
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

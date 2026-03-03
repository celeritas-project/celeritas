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
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalModel.hh"
#include "celeritas/optical/Types.hh"

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

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with reference to data being loaded.
 */
GeantPhysicsLoader::GeantPhysicsLoader(ImportData& imported,
                                       GeoOpticalIdMap const& optical_ids)
    : imported_{imported}, import_optical_model_{optical_ids}
{
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
        return true;
    }

    using MemberFuncPtr = void (GeantPhysicsLoader::*)(G4VProcess const&);
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
    CELER_LOG(debug) << "Loading process " << name << "(\""
                     << p.GetProcessName() << "\")";
    (this->*mfptr)(p);
    return true;
}

//---------------------------------------------------------------------------//
//! Activate Cherenkov emission
void GeantPhysicsLoader::cerenkov(G4VProcess const&)
{
    imported_.optical_physics.cherenkov = true;
}

//---------------------------------------------------------------------------//
//! Initialize muon-catalyzed fusion
void GeantPhysicsLoader::muon_minus_atomic_capture(G4VProcess const&)
{
    // G4MuonMinusAtomicCapture is a G4ProcessType::fHadronic
    // It is also a G4VRestProcess and does not require import data
    imported_.mucf_physics = inp::MucfPhysics::from_default();
}

//---------------------------------------------------------------------------//
//! Activate optical scintillation
void GeantPhysicsLoader::scintillation(G4VProcess const&)
{
    imported_.optical_physics.scintillation = true;
}

//---------------------------------------------------------------------------//
//! Activate optical absorption
void GeantPhysicsLoader::op_absorption(G4VProcess const&)
{
    if (!import_optical_model_)
    {
        CELER_LOG(warning) << "Ignoring inapplicable G4OpAbsorption";
        return;
    }

    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::absorption));
}

//---------------------------------------------------------------------------//
//! Activate optical surface physics
void GeantPhysicsLoader::op_boundary(G4VProcess const&)
{
    auto& surfaces = imported_.optical_physics.surfaces;

    // Load each geometry surface and print any errors that occur
    {
        auto& materials = imported_.optical_materials;
        if (materials.empty())
        {
            CELER_LOG(error) << "Optical boundary process is defined but no "
                                "optical materials are present";
            return;
        }

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

    CELER_LOG(debug) << "Loaded " << surfaces.materials.size()
                     << " optical surfaces (" << num_phys_surfaces
                     << " physics surfaces)";
    CELER_ENSURE(surfaces);
}

//---------------------------------------------------------------------------//
//! Activate Mie scattering
void GeantPhysicsLoader::op_mie_hg(G4VProcess const&)
{
    if (!import_optical_model_)
    {
        CELER_LOG(warning) << "Ignoring inapplicable G4OpMieHG";
        return;
    }

    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::mie));
}

//---------------------------------------------------------------------------//
//! Activate rayleigh scattering
void GeantPhysicsLoader::op_rayleigh(G4VProcess const&)
{
    if (!import_optical_model_)
    {
        CELER_LOG(warning) << "Ignoring inapplicable G4OpRayleigh";
        return;
    }

    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::rayleigh));
}

//---------------------------------------------------------------------------//
//! Activate wavelength shifting
void GeantPhysicsLoader::op_wls(G4VProcess const&)
{
    if (!import_optical_model_)
    {
        CELER_LOG(warning) << "Ignoring inapplicable G4OpWLS";
        return;
    }

    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::wls));

#if G4VERSION_NUMBER >= 1070
    // Save time profile
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    imported_.optical_params.wls_time_profile
        = geant_to_wls_distribution(params->GetWLSTimeProfile());
#endif
}

//---------------------------------------------------------------------------//
//! Activate wavelength shifting additional distribution
void GeantPhysicsLoader::op_wls2(G4VProcess const&)
{
    if (!import_optical_model_)
    {
        CELER_LOG(warning) << "Ignoring inapplicable G4OpWLS2";
        return;
    }

#if G4VERSION_NUMBER >= 1070
    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::wls2));

    // Save time profile
    auto* params = G4OpticalParameters::Instance();
    CELER_ASSERT(params);
    imported_.optical_params.wls_time_profile
        = geant_to_wls_distribution(params->GetWLS2TimeProfile());
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

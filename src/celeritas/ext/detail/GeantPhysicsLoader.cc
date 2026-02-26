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
#endif

#include "corecel/io/Logger.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "celeritas/inp/MucfPhysics.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalModel.hh"

namespace celeritas
{
namespace detail
{
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
        GPL_TYPE_FUNC(G4MuonMinusAtomicCapture, mucf),
        GPL_TYPE_FUNC(G4Cerenkov,              cerenkov),
        GPL_TYPE_FUNC(G4Scintillation,         scintillation),
        GPL_TYPE_FUNC(G4OpAbsorption,          absorption),
        GPL_TYPE_FUNC(G4OpBoundaryProcess,     boundary),
        GPL_TYPE_FUNC(G4OpMieHG,               mie),
        GPL_TYPE_FUNC(G4OpRayleigh,            rayleigh),
        GPL_TYPE_FUNC(G4OpWLS,                 wls),
#if G4VERSION_NUMBER >= 1070
        GPL_TYPE_FUNC(G4OpWLS2,                wls2),
#endif
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
void GeantPhysicsLoader::mucf(G4VProcess const&)
{
    // G4MuonMinusAtomicCapture is a G4ProcessType::fHadronic
    // It is also a G4VRestProcess and does not require import data
    imported_.mucf_physics = inp::MucfPhysics::from_default();
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::cerenkov(G4VProcess const&)
{
    imported_.optical_physics.cherenkov = true;
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::scintillation(G4VProcess const&)
{
    imported_.optical_physics.scintillation = true;
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::absorption(G4VProcess const&)
{
    CELER_EXPECT(import_optical_model_);
    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::absorption));
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::rayleigh(G4VProcess const&)
{
    CELER_EXPECT(import_optical_model_);
    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::rayleigh));
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::wls(G4VProcess const&)
{
    CELER_EXPECT(import_optical_model_);
    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::wls));
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::mie(G4VProcess const&)
{
    CELER_EXPECT(import_optical_model_);
    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::mie));
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::boundary(G4VProcess const&)
{
    // Surface physics importing is handled separately
}

//---------------------------------------------------------------------------//
void GeantPhysicsLoader::wls2(G4VProcess const&)
{
#if G4VERSION_NUMBER >= 1070
    CELER_EXPECT(import_optical_model_);
    imported_.optical_models.push_back(
        import_optical_model_(optical::ImportModelClass::wls2));
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

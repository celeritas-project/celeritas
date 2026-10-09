//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/BuildOutput.cc
//---------------------------------------------------------------------------//
#include "BuildOutput.hh"

#include <string>
#include <string_view>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"
#include "corecel/Version.hh"

#include "corecel/Macros.hh"

#include "JsonPimpl.hh"
#include "StringUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void BuildOutput::output(JsonPimpl* j) const
{
    auto obj = nlohmann::json::object({
        {"version", std::string(version_string)},
    });

    obj["config"] = [] {
        auto cfg = nlohmann::json::object();

        cfg["use"] = [] {
            std::vector<std::string> options;
#define CO_ADD_OPT(NAME) \
    if constexpr (CELERITAS_USE_##NAME) \
    { \
        options.push_back(celeritas::tolower(#NAME)); \
    }
            CO_ADD_OPT(COVFIE);
            CO_ADD_OPT(CUDA);
            CO_ADD_OPT(GEANT4);
            CO_ADD_OPT(HEPMC3);
            CO_ADD_OPT(HIP);
            CO_ADD_OPT(LARSOFT);
            CO_ADD_OPT(MPI);
            CO_ADD_OPT(OPENMP);
            CO_ADD_OPT(PERFETTO);
            CO_ADD_OPT(ROOT);
            CO_ADD_OPT(VECGEOM);
#undef CO_ADD_OPT
            return options;
        }();

#define CO_ADD_CFG(NAME) cfg[#NAME] = std::string(config::NAME);
        CO_ADD_CFG(build_type);
        CO_ADD_CFG(hostname);
        CO_ADD_CFG(real_type);
        CO_ADD_CFG(units);
        CO_ADD_CFG(constants);
        CO_ADD_CFG(openmp);
        CO_ADD_CFG(core_geo);
        CO_ADD_CFG(core_rng);
        CO_ADD_CFG(gpu_architectures);
        CO_ADD_CFG(reseed);
#undef CO_ADD_CFG
        if constexpr (CELER_USE_DEVICE)
        {
            // Be specific about host/device debug options
            std::vector<std::string> v;
            if (CELERITAS_DEBUG)
                v.push_back("host");
            if (CELERITAS_DEVICE_DEBUG)
                v.push_back("device");
            cfg["debug"] = std::move(v);
        }
        else
        {
            // Only a boolean option is needed
            cfg["debug"] = bool(CELERITAS_DEBUG);
        }

        cfg["versions"] = [] {
            auto deps = nlohmann::json::object();

            // TODO: export CELERITAS_ENABLED_COMPONENTS from cmake
            auto append_version = [&deps](bool enabled, std::string_view name) {
                if (!enabled)
                    return;
                auto lower = tolower(name);
                char const* v = package_version_cstring(lower.c_str());
                CELER_VALIDATE(v != nullptr,
                               << "invalid package '" << name << "'");
                deps[std::string{name}] = std::string{v};
            };
            append_version(CELERITAS_USE_COVFIE, "covfie");
            append_version(CELERITAS_USE_CUDA, "CUDA");
            append_version(CELERITAS_USE_CUDA, "Thrust");
            append_version(CELERITAS_USE_GEANT4, "CLHEP");
            append_version(CELERITAS_USE_GEANT4, "Geant4");
            append_version(CELERITAS_USE_HEPMC3, "HepMC3");
            append_version(CELERITAS_USE_HIP, "hip");
            append_version(CELERITAS_USE_HIP, "hipcub");
            append_version(CELERITAS_USE_HIP, "hiprand");
            append_version(CELERITAS_USE_HIP, "roctracer");
            append_version(CELERITAS_USE_LARSOFT, "LArSoft");
            append_version(CELERITAS_USE_ROOT, "ROOT");
            append_version(CELERITAS_USE_VECGEOM, "G4VG");
            append_version(CELERITAS_USE_VECGEOM || CELERITAS_GEANT4_USOLIDS,
                           "VecGeom");

            return deps;
        }();

        if constexpr (CELERITAS_USE_GEANT4)
        {
            cfg["geant4"] = std::string(config::geant4_options);
        }

        if constexpr (CELERITAS_USE_VECGEOM || CELERITAS_GEANT4_USOLIDS)
        {
            cfg["vecgeom"] = std::string(config::vecgeom_options);
        }

        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            cfg["orange_torus"] = static_cast<bool>(CELERITAS_ORANGE_TORUS);
        }

        return cfg;
    }();

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

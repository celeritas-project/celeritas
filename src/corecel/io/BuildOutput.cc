//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/BuildOutput.cc
//---------------------------------------------------------------------------//
#include "BuildOutput.hh"

#include <string>
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
#define CO_ADD_OPT(NAME)                              \
    if constexpr (CELERITAS_USE_##NAME)               \
    {                                                 \
        options.push_back(celeritas::tolower(#NAME)); \
    }
            CO_ADD_OPT(CUDA);
            CO_ADD_OPT(GEANT4);
            CO_ADD_OPT(HEPMC3);
            CO_ADD_OPT(HIP);
            CO_ADD_OPT(MPI);
            CO_ADD_OPT(OPENMP);
            CO_ADD_OPT(ROOT);
            CO_ADD_OPT(VECGEOM);
#undef CO_ADD_OPT
            return options;
        }();

#define CO_ADD_CFG(NAME) cfg[#NAME] = std::string(cmake::NAME);
        CO_ADD_CFG(build_type);
        CO_ADD_CFG(hostname);
        CO_ADD_CFG(real_type);
        CO_ADD_CFG(units);
        CO_ADD_CFG(openmp);
        CO_ADD_CFG(core_geo);
        CO_ADD_CFG(core_rng);
        CO_ADD_CFG(gpu_architectures);
#undef CO_ADD_CFG
        cfg["debug"] = bool(CELERITAS_DEBUG);

        cfg["versions"] = [] {
            auto deps = nlohmann::json::object();

#define CO_ADD_COND_VERS(USE, NAME, LOWER)                 \
    if constexpr (CELERITAS_USE_##USE)                     \
    {                                                      \
        deps[#NAME] = std::string(cmake::LOWER##_version); \
    }
            CO_ADD_COND_VERS(GEANT4, CLHEP, clhep);
            CO_ADD_COND_VERS(GEANT4, Geant4, geant4);
            CO_ADD_COND_VERS(CUDA, CUDA, cuda);
            CO_ADD_COND_VERS(CUDA, Thrust, thrust);
            CO_ADD_COND_VERS(HEPMC3, HepMC3, hepmc3);
            CO_ADD_COND_VERS(HIP, HIP, hip);
            CO_ADD_COND_VERS(ROOT, ROOT, root);
            CO_ADD_COND_VERS(VECGEOM, VecGeom, vecgeom);
#undef CO_ADD_COND_VERS
            if constexpr (CELERITAS_USE_VECGEOM)
            {
                deps["vecgeom_options"] = std::string(cmake::vecgeom_options);
            }
            return deps;
        }();

        return cfg;
    }();

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

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
        {"version", std::string(celeritas_version)},
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

#define CO_ADD_CFG(NAME) cfg[#NAME] = celeritas_##NAME;
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

            if constexpr (CELERITAS_USE_GEANT4)
            {
                deps["CLHEP"] = celeritas_clhep_version;
                deps["Geant4"] = celeritas_geant4_version;
            }
            if constexpr (CELERITAS_USE_CUDA)
            {
                deps["CUDA"] = celeritas_cuda_version;
                deps["Thrust"] = celeritas_thrust_version;
            }
            if constexpr (CELERITAS_USE_HEPMC3)
            {
                deps["HepMC3"] = celeritas_hepmc3_version;
            }
            if constexpr (CELERITAS_USE_HIP)
            {
                deps["HIP"] = celeritas_hip_version;
            }
            if constexpr (CELERITAS_USE_ROOT)
            {
                deps["ROOT"] = celeritas_root_version;
            }
            if constexpr (CELERITAS_USE_VECGEOM)
            {
                deps["VecGeom"] = celeritas_vecgeom_version;
            }
            return deps;
        }();

        return cfg;
    }();

    // DEPRECATED: remove in v0.6
    {
        auto& cfg = obj["config"];

#define CO_SAVE_CFG(NAME) cfg[#NAME] = bool(NAME)
        CO_SAVE_CFG(CELERITAS_USE_CUDA);
        CO_SAVE_CFG(CELERITAS_USE_GEANT4);
        CO_SAVE_CFG(CELERITAS_USE_HEPMC3);
        CO_SAVE_CFG(CELERITAS_USE_HIP);
        CO_SAVE_CFG(CELERITAS_USE_MPI);
        CO_SAVE_CFG(CELERITAS_USE_OPENMP);
        CO_SAVE_CFG(CELERITAS_USE_ROOT);
        CO_SAVE_CFG(CELERITAS_USE_VECGEOM);
        CO_SAVE_CFG(CELERITAS_DEBUG);
#undef CO_SAVE_CFG
        cfg["CELERITAS_BUILD_TYPE"] = celeritas_build_type;
        cfg["CELERITAS_HOSTNAME"] = celeritas_hostname;
        cfg["CELERITAS_REAL_TYPE"] = celeritas_real_type;
        cfg["CELERITAS_CORE_GEO"] = celeritas_core_geo;
        cfg["CELERITAS_CORE_RNG"] = celeritas_core_rng;
        cfg["CELERITAS_UNITS"] = celeritas_units;
        if constexpr (CELERITAS_USE_GEANT4)
        {
            cfg["CLHEP_VERSION"] = celeritas_clhep_version;
            cfg["Geant4_VERSION"] = celeritas_geant4_version;
        }
        if constexpr (CELERITAS_USE_CUDA)
        {
            cfg["CUDA_VERSION"] = celeritas_cuda_version;
            cfg["Thrust_VERSION"] = celeritas_thrust_version;
        }
        if constexpr (CELERITAS_USE_HIP)
        {
            cfg["HIP_VERSION"] = celeritas_hip_version;
        }
        if constexpr (CELERITAS_USE_VECGEOM)
        {
            cfg["VecGeom_VERSION"] = celeritas_vecgeom_version;
        }
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

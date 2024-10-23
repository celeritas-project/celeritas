//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void BuildOutput::output(JsonPimpl* j) const
{
    auto obj = nlohmann::json::object();

    obj["version"] = std::string(celeritas_version);

    {
        auto cfg = nlohmann::json::object();
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

        obj["config"] = std::move(cfg);
    }

    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

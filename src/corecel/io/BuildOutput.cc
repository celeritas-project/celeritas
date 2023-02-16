//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/BuildOutput.cc
//---------------------------------------------------------------------------//
#include "BuildOutput.hh"

#include <string>
#include <utility>

#include "celeritas_cmake_strings.h"
#include "celeritas_config.h"
#include "celeritas_version.h"

#include "JsonPimpl.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void BuildOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    auto obj = nlohmann::json::object();

    obj["version"] = std::string(celeritas_version);

    {
        auto cfg = nlohmann::json::object();
#    define CO_SAVE_CFG(NAME) cfg[#NAME] = bool(NAME)
        CO_SAVE_CFG(CELERITAS_USE_CUDA);
        CO_SAVE_CFG(CELERITAS_USE_GEANT4);
        CO_SAVE_CFG(CELERITAS_USE_HEPMC3);
        CO_SAVE_CFG(CELERITAS_USE_HIP);
        CO_SAVE_CFG(CELERITAS_USE_JSON);
        CO_SAVE_CFG(CELERITAS_USE_MPI);
        CO_SAVE_CFG(CELERITAS_USE_OPENMP);
        CO_SAVE_CFG(CELERITAS_USE_ROOT);
        CO_SAVE_CFG(CELERITAS_USE_VECGEOM);
        CO_SAVE_CFG(CELERITAS_DEBUG);
        CO_SAVE_CFG(CELERITAS_LAUNCH_BOUNDS);
#    undef CO_SAVE_CFG
        cfg["CELERITAS_BUILD_TYPE"] = celeritas_build_type;
        cfg["CELERITAS_HOSTNAME"] = celeritas_hostname;
        cfg["CELERITAS_RNG"] = celeritas_rng;
        if constexpr (CELERITAS_USE_GEANT4)
        {
            cfg["CLHEP_VERSION"] = celeritas_clhep_version;
            cfg["Geant4_VERSION"] = celeritas_geant4_version;
        }
        if constexpr (CELERITAS_USE_VECGEOM)
        {
            cfg["VecGeom_VERSION"] = celeritas_vecgeom_version;
        }

        obj["config"] = std::move(cfg);
    }

    j->obj = std::move(obj);
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

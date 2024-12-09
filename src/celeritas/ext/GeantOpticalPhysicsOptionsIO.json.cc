//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantOpticalPhysicsOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "GeantOpticalPhysicsOptionsIO.json.hh"

#include <string>

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/StringEnumMapper.hh"

#include "GeantOpticalPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
static char const format_str[] = "geant4-optical-physics";

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, WLSTimeProfileSelection& value)
{
    static auto const from_string
        = StringEnumMapper<WLSTimeProfileSelection>::from_cstring_func(
            to_cstring, "wls time profile");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, WLSTimeProfileSelection const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, CherenkovPhysicsOptions& options)
{
    if (j.is_null())
    {
        // Null json means deactivated process
        options = CherenkovPhysicsOptions{};
        options.enable = false;
        return;
    }
#define GCPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GCPO_LOAD_OPTION(enable);
    GCPO_LOAD_OPTION(stack_photons);
    GCPO_LOAD_OPTION(track_secondaries_first);
    GCPO_LOAD_OPTION(max_photons);
    GCPO_LOAD_OPTION(max_beta_change);
#undef GCPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, CherenkovPhysicsOptions const& inp)
{
    if (!inp)
    {
        // Special case for process being inactivated
        j = nullptr;
        return;
    }

    j = {
        CELER_JSON_PAIR(inp, enable),
        CELER_JSON_PAIR(inp, stack_photons),
        CELER_JSON_PAIR(inp, track_secondaries_first),
        CELER_JSON_PAIR(inp, max_photons),
        CELER_JSON_PAIR(inp, max_beta_change),
    };
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, ScintillationPhysicsOptions& options)
{
    if (j.is_null())
    {
        // Null json means deactivated process
        options = ScintillationPhysicsOptions{};
        options.enable = false;
        return;
    }
#define GSPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GSPO_LOAD_OPTION(enable);
    GSPO_LOAD_OPTION(stack_photons);
    GSPO_LOAD_OPTION(track_secondaries_first);
    GSPO_LOAD_OPTION(by_particle_type);
    GSPO_LOAD_OPTION(finite_rise_time);
    GSPO_LOAD_OPTION(track_info);
#undef GSPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, ScintillationPhysicsOptions const& inp)
{
    if (!inp)
    {
        // Special case for process being inactivated
        j = nullptr;
        return;
    }

    j = {
        CELER_JSON_PAIR(inp, enable),
        CELER_JSON_PAIR(inp, stack_photons),
        CELER_JSON_PAIR(inp, track_secondaries_first),
        CELER_JSON_PAIR(inp, by_particle_type),
        CELER_JSON_PAIR(inp, finite_rise_time),
        CELER_JSON_PAIR(inp, track_info),
    };
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, BoundaryPhysicsOptions& options)
{
    if (j.is_null())
    {
        // Null json means deactivated process
        options = BoundaryPhysicsOptions{};
        options.enable = false;
        return;
    }
#define GBPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GBPO_LOAD_OPTION(enable);
    GBPO_LOAD_OPTION(invoke_sd);
#undef GBPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, BoundaryPhysicsOptions const& inp)
{
    if (!inp)
    {
        // Special case for process being inactivated
        j = nullptr;
        return;
    }

    j = {
        CELER_JSON_PAIR(inp, enable),
        CELER_JSON_PAIR(inp, invoke_sd),
    };
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, GeantOpticalPhysicsOptions& options)
{
    if (j.is_null())
    {
        // Null json means deactivated options
        options = GeantOpticalPhysicsOptions::deactivated();
        return;
    }

#define GOPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    check_format(j, format_str);
    GOPO_LOAD_OPTION(cherenkov);
    GOPO_LOAD_OPTION(scintillation);
    GOPO_LOAD_OPTION(wavelength_shifting);
    GOPO_LOAD_OPTION(wavelength_shifting2);
    GOPO_LOAD_OPTION(boundary);
    GOPO_LOAD_OPTION(absorption);
    GOPO_LOAD_OPTION(rayleigh_scattering);
    GOPO_LOAD_OPTION(mie_scattering);
    GOPO_LOAD_OPTION(verbose);
#undef GOPO_LOAD_OPTION
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, GeantOpticalPhysicsOptions const& inp)
{
    if (!inp)
    {
        // Special case for all processes being inactivated
        j = nullptr;
        return;
    }

    j = {
        CELER_JSON_PAIR(inp, cherenkov),
        CELER_JSON_PAIR(inp, scintillation),
        CELER_JSON_PAIR(inp, wavelength_shifting),
        CELER_JSON_PAIR(inp, wavelength_shifting2),
        CELER_JSON_PAIR(inp, boundary),
        CELER_JSON_PAIR(inp, absorption),
        CELER_JSON_PAIR(inp, rayleigh_scattering),
        CELER_JSON_PAIR(inp, mie_scattering),
        CELER_JSON_PAIR(inp, verbose),
    };

    save_format(j, format_str);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

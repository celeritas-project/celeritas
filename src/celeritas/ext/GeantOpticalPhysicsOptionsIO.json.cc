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
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, GeantOpticalPhysicsOptions& options)
{
#define GOPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    check_format(j, format_str);
    GOPO_LOAD_OPTION(cerenkov_radiation);
    GOPO_LOAD_OPTION(scintillation);
    GOPO_LOAD_OPTION(wavelength_shifting);
    GOPO_LOAD_OPTION(wavelength_shifting2);
    GOPO_LOAD_OPTION(boundary);
    GOPO_LOAD_OPTION(absorption);
    GOPO_LOAD_OPTION(rayleigh_scattering);
    GOPO_LOAD_OPTION(mie_scattering);

    GOPO_LOAD_OPTION(cerenkov_stack_photons);
    GOPO_LOAD_OPTION(cerenkov_track_secondaries_first);
    GOPO_LOAD_OPTION(cerenkov_max_photons);
    GOPO_LOAD_OPTION(cerenkov_max_beta_change);

    GOPO_LOAD_OPTION(scint_stack_photons);
    GOPO_LOAD_OPTION(scint_track_secondaries_first);
    GOPO_LOAD_OPTION(scint_by_particle_type);
    GOPO_LOAD_OPTION(scint_finite_rise_time);
    GOPO_LOAD_OPTION(scint_track_info);

    GOPO_LOAD_OPTION(invoke_sd);

    GOPO_LOAD_OPTION(verbose);
#undef GOPO_LOAD_OPTION
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, GeantOpticalPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, cerenkov_radiation),
        CELER_JSON_PAIR(inp, scintillation),
        CELER_JSON_PAIR(inp, wavelength_shifting),
        CELER_JSON_PAIR(inp, wavelength_shifting2),
        CELER_JSON_PAIR(inp, boundary),
        CELER_JSON_PAIR(inp, absorption),
        CELER_JSON_PAIR(inp, rayleigh_scattering),
        CELER_JSON_PAIR(inp, mie_scattering),

        CELER_JSON_PAIR(inp, cerenkov_stack_photons),
        CELER_JSON_PAIR(inp, cerenkov_track_secondaries_first),
        CELER_JSON_PAIR(inp, cerenkov_max_photons),
        CELER_JSON_PAIR(inp, cerenkov_max_beta_change),

        CELER_JSON_PAIR(inp, scint_stack_photons),
        CELER_JSON_PAIR(inp, scint_track_secondaries_first),
        CELER_JSON_PAIR(inp, scint_by_particle_type),
        CELER_JSON_PAIR(inp, scint_finite_rise_time),
        CELER_JSON_PAIR(inp, scint_track_info),

        CELER_JSON_PAIR(inp, invoke_sd),

        CELER_JSON_PAIR(inp, verbose),
    };

    save_format(j, format_str);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantOpticalPhysicsOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "GeantOpticalPhysicsOptionsIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "celeritas/TypesIO.json.hh"

#include "GeantOpticalPhysicsOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
static char const format_str[] = "geant4-optical-physics";

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, CherenkovPhysicsOptions& options)
{
#define GCPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GCPO_LOAD_OPTION(stack_photons);
    GCPO_LOAD_OPTION(track_secondaries_first);
    GCPO_LOAD_OPTION(max_photons);
    GCPO_LOAD_OPTION(max_beta_change);
#undef GCPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, CherenkovPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, stack_photons),
        CELER_JSON_PAIR(inp, track_secondaries_first),
        CELER_JSON_PAIR(inp, max_photons),
        CELER_JSON_PAIR(inp, max_beta_change),
    };
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, ScintillationPhysicsOptions& options)
{
#define GSPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GSPO_LOAD_OPTION(stack_photons);
    GSPO_LOAD_OPTION(track_secondaries_first);
    GSPO_LOAD_OPTION(by_particle_type);
    GSPO_LOAD_OPTION(finite_rise_time);
    GSPO_LOAD_OPTION(track_info);
#undef GSPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, ScintillationPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, stack_photons),
        CELER_JSON_PAIR(inp, track_secondaries_first),
        CELER_JSON_PAIR(inp, by_particle_type),
        CELER_JSON_PAIR(inp, finite_rise_time),
        CELER_JSON_PAIR(inp, track_info),
    };
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, WavelengthShiftingOptions& options)
{
    CELER_JSON_LOAD_OPTION(j, options, time_profile);
}

void to_json(nlohmann::json& j, WavelengthShiftingOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, time_profile),
    };
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, BoundaryPhysicsOptions& options)
{
#define GBPO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    GBPO_LOAD_OPTION(invoke_sd);
#undef GBPO_LOAD_OPTION
}

void to_json(nlohmann::json& j, BoundaryPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, invoke_sd),
    };
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, GeantOpticalPhysicsOptions& options)
{
    check_format(j, format_str);

    CELER_JSON_LOAD_OPTIONAL(j, options, cherenkov);
    CELER_JSON_LOAD_OPTIONAL(j, options, scintillation);
    CELER_JSON_LOAD_OPTIONAL(j, options, wavelength_shifting);
    CELER_JSON_LOAD_OPTIONAL(j, options, wavelength_shifting2);
    CELER_JSON_LOAD_OPTIONAL(j, options, boundary);
    CELER_JSON_LOAD_OPTION(j, options, absorption);
    CELER_JSON_LOAD_OPTION(j, options, rayleigh_scattering);
    CELER_JSON_LOAD_OPTION(j, options, mie_scattering);
    CELER_JSON_LOAD_OPTION(j, options, verbose);
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, GeantOpticalPhysicsOptions const& inp)
{
    j = {
        CELER_JSON_PAIR_OPTIONAL(inp, cherenkov),
        CELER_JSON_PAIR_OPTIONAL(inp, scintillation),
        CELER_JSON_PAIR_OPTIONAL(inp, wavelength_shifting),
        CELER_JSON_PAIR_OPTIONAL(inp, wavelength_shifting2),
        CELER_JSON_PAIR_OPTIONAL(inp, boundary),
        CELER_JSON_PAIR(inp, absorption),
        CELER_JSON_PAIR(inp, rayleigh_scattering),
        CELER_JSON_PAIR(inp, mie_scattering),
        CELER_JSON_PAIR(inp, verbose),
    };

    save_format(j, format_str);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

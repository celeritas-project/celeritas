//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RunInputIO.json.hh"

#include "corecel/io/StringEnumMapper.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/phys/PrimaryGeneratorOptionsIO.json.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, RunInput& v)
{
#define RI_LOAD_OPTION(NAME)            \
    do                                  \
    {                                   \
        if (j.contains(#NAME))          \
            j.at(#NAME).get_to(v.NAME); \
    } while (0)

    RI_LOAD_OPTION(event_file);
    RI_LOAD_OPTION(primary_options);

    RI_LOAD_OPTION(physics_list);
    RI_LOAD_OPTION(physics_options);

    RI_LOAD_OPTION(field_options);

    RI_LOAD_OPTION(step_diagnostic);
    RI_LOAD_OPTION(step_diagnostic_bins);

#undef RI_LOAD_OPTION

    CELER_VALIDATE(v.event_file.empty() != !v.primary_options,
                   << "either a HepMC3 filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(v.physics_list == PhysicsList::geant_physics_list
                       || !j.contains("physics_options"),
                   << "'physics_options' can only be specified for "
                      "'geant_physics_list'");
}

//---------------------------------------------------------------------------//
/*!
 * Save options to JSON.
 */
void to_json(nlohmann::json& j, RunInput const& v)
{
    j = nlohmann::json::object();
    RunInput const default_args;
#define RI_SAVE_OPTION(NAME)                \
    do                                      \
    {                                       \
        if (!(v.NAME == default_args.NAME)) \
            j[#NAME] = v.NAME;              \
    } while (0)
#define RI_SAVE_REQUIRED(NAME) j[#NAME] = v.NAME

    RI_SAVE_OPTION(event_file);
    if (v.event_file.empty())
    {
        RI_SAVE_REQUIRED(primary_options);
    }

    RI_SAVE_OPTION(physics_list);
    if (v.physics_list == PhysicsList::geant_physics_list)
    {
        RI_SAVE_OPTION(physics_options);
    }

    RI_SAVE_OPTION(field_options);

    RI_SAVE_OPTION(step_diagnostic);
    RI_SAVE_OPTION(step_diagnostic_bins);

#undef RI_SAVE_OPTION
#undef RI_SAVE_REQUIRED
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

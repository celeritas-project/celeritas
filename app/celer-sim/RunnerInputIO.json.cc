//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RunnerInputIO.json.hh"

#include <string>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/phys/PrimaryGeneratorOptionsIO.json.hh"
#include "celeritas/user/RootStepWriterIO.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, TrackOrder& value)
{
    static auto const from_string
        = StringEnumMapper<TrackOrder>::from_cstring_func(to_cstring,
                                                          "track order");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, TrackOrder const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 *
 * TODO: for version 1.0, remove deprecated options.
 */
void from_json(nlohmann::json const& j, RunnerInput& v)
{
#define LDIO_LOAD_OPTION(NAME)          \
    do                                  \
    {                                   \
        if (j.contains(#NAME))          \
            j.at(#NAME).get_to(v.NAME); \
    } while (0)
#define LDIO_LOAD_DEPRECATED(OLD, NEW)                                        \
    do                                                                        \
    {                                                                         \
        if (j.contains(#OLD))                                                 \
        {                                                                     \
            CELER_LOG(warning) << "Deprecated option '" << #OLD << "': use '" \
                               << #NEW << "' instead";                        \
            j.at(#OLD).get_to(v.NEW);                                         \
        }                                                                     \
    } while (0)
#define LDIO_LOAD_REQUIRED(NAME) j.at(#NAME).get_to(v.NAME)

    LDIO_LOAD_OPTION(cuda_heap_size);
    LDIO_LOAD_OPTION(cuda_stack_size);
    LDIO_LOAD_OPTION(environ);

    LDIO_LOAD_DEPRECATED(hepmc3_filename, event_filename);

    LDIO_LOAD_REQUIRED(geometry_filename);
    LDIO_LOAD_OPTION(physics_filename);
    LDIO_LOAD_OPTION(event_filename);

    LDIO_LOAD_OPTION(primary_gen_options);

    LDIO_LOAD_OPTION(mctruth_filename);
    LDIO_LOAD_OPTION(mctruth_filter);
    LDIO_LOAD_OPTION(simple_calo);
    LDIO_LOAD_OPTION(action_diagnostic);
    LDIO_LOAD_OPTION(step_diagnostic);
    LDIO_LOAD_OPTION(step_diagnostic_maxsteps);

    LDIO_LOAD_DEPRECATED(max_num_tracks, num_track_slots);

    LDIO_LOAD_OPTION(seed);
    LDIO_LOAD_OPTION(num_track_slots);
    LDIO_LOAD_OPTION(max_steps);
    LDIO_LOAD_REQUIRED(initializer_capacity);
    LDIO_LOAD_REQUIRED(max_events);
    LDIO_LOAD_REQUIRED(secondary_stack_factor);
    LDIO_LOAD_REQUIRED(use_device);
    LDIO_LOAD_OPTION(sync);
    LDIO_LOAD_OPTION(merge_events);
    LDIO_LOAD_OPTION(default_stream);

    LDIO_LOAD_OPTION(mag_field);
    LDIO_LOAD_OPTION(field_options);

    LDIO_LOAD_OPTION(step_limiter);
    LDIO_LOAD_OPTION(brem_combined);
    LDIO_LOAD_OPTION(track_order);
    LDIO_LOAD_OPTION(geant_options);

#undef LDIO_LOAD_OPTION
#undef LDIO_LOAD_REQUIRED

    CELER_VALIDATE(v.event_filename.empty() != !v.primary_gen_options,
                   << "either a event filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(!v.mctruth_filter || !v.mctruth_filename.empty(),
                   << "'mctruth_filter' cannot be specified without providing "
                      "'mctruth_filename'");
    CELER_VALIDATE(v.mag_field != RunnerInput::no_field()
                       || !j.contains("field_options"),
                   << "'field_options' cannot be specified without providing "
                      "'mag_field'");
}

//---------------------------------------------------------------------------//
/*!
 * Save options to JSON.
 */
void to_json(nlohmann::json& j, RunnerInput const& v)
{
    j = nlohmann::json::object();
    RunnerInput const default_args;
#define LDIO_SAVE_OPTION(NAME)           \
    do                                   \
    {                                    \
        if (v.NAME != default_args.NAME) \
            j[#NAME] = v.NAME;           \
    } while (0)
#define LDIO_SAVE_REQUIRED(NAME) j[#NAME] = v.NAME

    LDIO_SAVE_OPTION(cuda_heap_size);
    LDIO_SAVE_OPTION(cuda_stack_size);
    LDIO_SAVE_REQUIRED(environ);

    LDIO_SAVE_REQUIRED(geometry_filename);
    LDIO_SAVE_REQUIRED(physics_filename);
    LDIO_SAVE_OPTION(event_filename);
    if (v.event_filename.empty())
    {
        LDIO_SAVE_REQUIRED(primary_gen_options);
    }

    LDIO_SAVE_OPTION(mctruth_filename);
    if (!v.mctruth_filename.empty())
    {
        LDIO_SAVE_REQUIRED(mctruth_filter);
    }
    LDIO_SAVE_OPTION(simple_calo);
    LDIO_SAVE_OPTION(action_diagnostic);
    LDIO_SAVE_OPTION(step_diagnostic);
    LDIO_SAVE_OPTION(step_diagnostic_maxsteps);

    LDIO_SAVE_OPTION(seed);
    LDIO_SAVE_OPTION(num_track_slots);
    LDIO_SAVE_OPTION(max_steps);
    LDIO_SAVE_REQUIRED(initializer_capacity);
    LDIO_SAVE_REQUIRED(max_events);
    LDIO_SAVE_REQUIRED(secondary_stack_factor);
    LDIO_SAVE_REQUIRED(use_device);
    LDIO_SAVE_OPTION(sync);
    LDIO_SAVE_OPTION(merge_events);
    LDIO_SAVE_OPTION(default_stream);

    LDIO_SAVE_OPTION(mag_field);
    if (v.mag_field != RunnerInput::no_field())
    {
        LDIO_SAVE_REQUIRED(field_options);
    }

    LDIO_SAVE_OPTION(step_limiter);
    LDIO_SAVE_OPTION(brem_combined);

    LDIO_SAVE_OPTION(track_order);
    if (v.physics_filename.empty() || !ends_with(v.physics_filename, ".root"))
    {
        LDIO_SAVE_REQUIRED(geant_options);
    }

#undef LDIO_SAVE_OPTION
#undef LDIO_SAVE_REQUIRED
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

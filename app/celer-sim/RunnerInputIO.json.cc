//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RunnerInputIO.json.hh"

#include <string>

#include "corecel/Config.hh"

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/grid/GridTypesIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "celeritas/TypesIO.json.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/phys/PrimaryGeneratorOptionsIO.json.hh"
#include "celeritas/user/RootStepWriterIO.json.hh"

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
#define LDIO_LOAD_REQUIRED(NAME) CELER_JSON_LOAD_REQUIRED(j, v, NAME)
#define LDIO_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, v, NAME)
#define LDIO_LOAD_DEFAULT(NAME, DEFAULT) \
    CELER_JSON_LOAD_DEFAULT(j, v, NAME, DEFAULT)
#define LDIO_LOAD_DEPRECATED(OLD, NEW) \
    CELER_JSON_LOAD_DEPRECATED(j, v, OLD, NEW)

    // Check version (if available)
    check_format(j, "celer-sim");

    LDIO_LOAD_OPTION(cuda_heap_size);
    LDIO_LOAD_OPTION(cuda_stack_size);
    LDIO_LOAD_OPTION(environ);

    LDIO_LOAD_DEPRECATED(hepmc3_filename, event_file);
    LDIO_LOAD_DEPRECATED(event_filename, event_file);
    LDIO_LOAD_DEPRECATED(geometry_filename, geometry_file);
    LDIO_LOAD_DEPRECATED(physics_filename, physics_file);

    if (v.geometry_file.empty())
    {
        LDIO_LOAD_REQUIRED(geometry_file);
    }
    LDIO_LOAD_OPTION(physics_file);
    LDIO_LOAD_OPTION(event_file);

    LDIO_LOAD_OPTION(file_sampling_options);

    LDIO_LOAD_DEPRECATED(primary_gen_options, primary_options);

    LDIO_LOAD_OPTION(primary_options);

    LDIO_LOAD_DEPRECATED(mctruth_filename, mctruth_file);
    LDIO_LOAD_DEPRECATED(step_diagnostic_maxsteps, step_diagnostic_bins);

    LDIO_LOAD_OPTION(mctruth_file);
    LDIO_LOAD_OPTION(tracing_file);
    LDIO_LOAD_OPTION(mctruth_filter);
    LDIO_LOAD_OPTION(simple_calo);
    LDIO_LOAD_OPTION(action_diagnostic);
    LDIO_LOAD_OPTION(step_diagnostic);
    LDIO_LOAD_OPTION(step_diagnostic_bins);
    LDIO_LOAD_OPTION(slot_diagnostic_prefix);
    LDIO_LOAD_OPTION(write_track_counts);
    LDIO_LOAD_OPTION(write_step_times);
    LDIO_LOAD_OPTION(transporter_result);
    LDIO_LOAD_OPTION(status_checker);
    LDIO_LOAD_OPTION(log_progress);

    LDIO_LOAD_DEPRECATED(max_num_tracks, num_track_slots);
    LDIO_LOAD_DEPRECATED(sync, action_times);

    LDIO_LOAD_OPTION(seed);
    LDIO_LOAD_REQUIRED(use_device);

    // Get default capacities *integrated* over streams
    auto capacity = inp::CoreStateCapacity::from_default(v.use_device);

    LDIO_LOAD_DEFAULT(num_track_slots, capacity.tracks);
    LDIO_LOAD_OPTION(max_steps);
    LDIO_LOAD_DEFAULT(initializer_capacity, capacity.initializers);
    CELER_ASSERT(capacity.secondaries);
    LDIO_LOAD_DEFAULT(
        secondary_stack_factor,
        static_cast<real_type>(*capacity.secondaries) / capacity.tracks);
    LDIO_LOAD_OPTION(interpolation);
    LDIO_LOAD_OPTION(poly_spline_order);
    LDIO_LOAD_OPTION(action_times);
    LDIO_LOAD_OPTION(merge_events);
    if (j.count("default_stream"))
    {
        CELER_LOG(warning) << "Ignoring removed option 'default_stream'";
    }
    LDIO_LOAD_DEFAULT(warm_up, v.use_device ? true : false);

    LDIO_LOAD_DEPRECATED(mag_field, field);

    LDIO_LOAD_OPTION(field);
    LDIO_LOAD_OPTION(field_options);

    LDIO_LOAD_DEPRECATED(geant_options, physics_options);

    LDIO_LOAD_OPTION(step_limiter);
    LDIO_LOAD_DEFAULT(
        track_order, v.use_device ? TrackOrder::init_charge : TrackOrder::none);
    LDIO_LOAD_OPTION(physics_options);

    LDIO_LOAD_OPTION(optical);

#undef LDIO_LOAD_DEPRECATED
#undef LDIO_LOAD_OPTION
#undef LDIO_LOAD_REQUIRED

    CELER_VALIDATE(v.field != RunnerInput::no_field()
                       || !j.contains("field_options"),
                   << "'field_options' cannot be specified without providing "
                      "'field'");
}

//---------------------------------------------------------------------------//
/*!
 * Save options to JSON.
 */
void to_json(nlohmann::json& j, RunnerInput const& v)
{
#define LDIO_SAVE(NAME) CELER_JSON_SAVE(j, v, NAME)
#define LDIO_SAVE_WHEN(NAME, COND) CELER_JSON_SAVE_WHEN(j, v, NAME, COND)
#define LDIO_SAVE_OPTION(NAME) \
    LDIO_SAVE_WHEN(NAME, v.NAME != default_args.NAME)

    j = nlohmann::json::object();
    RunnerInput const default_args;

    // Save version and celer-sim format
    save_format(j, "celer-sim");

    LDIO_SAVE_OPTION(cuda_heap_size);
    LDIO_SAVE_OPTION(cuda_stack_size);
    LDIO_SAVE(environ);

    LDIO_SAVE(geometry_file);
    LDIO_SAVE(physics_file);
    LDIO_SAVE_OPTION(event_file);
    LDIO_SAVE_WHEN(file_sampling_options,
                   ends_with(v.event_file, ".root")
                       && static_cast<bool>(v.file_sampling_options));
    LDIO_SAVE_WHEN(primary_options, v.event_file.empty());

    LDIO_SAVE_OPTION(mctruth_file);
    LDIO_SAVE_WHEN(tracing_file, CELERITAS_USE_PERFETTO);
    LDIO_SAVE_WHEN(mctruth_filter, !v.mctruth_file.empty());
    LDIO_SAVE(simple_calo);
    LDIO_SAVE(action_diagnostic);
    LDIO_SAVE(step_diagnostic);
    LDIO_SAVE_OPTION(step_diagnostic_bins);
    LDIO_SAVE_OPTION(slot_diagnostic_prefix);
    LDIO_SAVE(write_track_counts);
    LDIO_SAVE(write_step_times);
    LDIO_SAVE(transporter_result);
    LDIO_SAVE(status_checker);
    LDIO_SAVE(log_progress);

    LDIO_SAVE(seed);
    LDIO_SAVE(num_track_slots);
    LDIO_SAVE_OPTION(max_steps);
    LDIO_SAVE(initializer_capacity);
    LDIO_SAVE(secondary_stack_factor);
    LDIO_SAVE_OPTION(interpolation);
    LDIO_SAVE_OPTION(poly_spline_order);
    LDIO_SAVE(use_device);
    LDIO_SAVE(action_times);
    LDIO_SAVE(merge_events);
    LDIO_SAVE(warm_up);

    LDIO_SAVE_OPTION(field);
    LDIO_SAVE_WHEN(field_options, v.field != RunnerInput::no_field());

    LDIO_SAVE_OPTION(step_limiter);

    LDIO_SAVE(track_order);
    LDIO_SAVE_WHEN(physics_options,
                   v.physics_file.empty()
                       || !ends_with(v.physics_file, ".root"));

    LDIO_SAVE_WHEN(optical, v.optical);

#undef LDIO_SAVE_OPTION
#undef LDIO_SAVE_WHEN
#undef LDIO_SAVE
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j,
               app::RunnerInput::EventFileSampling& efs)
{
    CELER_JSON_LOAD_REQUIRED(j, efs, num_events);
    CELER_JSON_LOAD_REQUIRED(j, efs, num_merged);
}

void to_json(nlohmann::json& j, app::RunnerInput::EventFileSampling const& efs)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(efs, num_events),
        CELER_JSON_PAIR(efs, num_merged),
    };
}

void from_json(nlohmann::json const& j, app::RunnerInput::OpticalOptions& oo)
{
    CELER_JSON_LOAD_REQUIRED(j, oo, num_track_slots);
    CELER_JSON_LOAD_REQUIRED(j, oo, buffer_capacity);
    CELER_JSON_LOAD_REQUIRED(j, oo, auto_flush);
    CELER_JSON_LOAD_OPTION(j, oo, max_steps);
    CELER_JSON_LOAD_OPTION(j, oo, cherenkov);
    CELER_JSON_LOAD_OPTION(j, oo, scintillation);
}

void to_json(nlohmann::json& j, app::RunnerInput::OpticalOptions const& oo)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(oo, num_track_slots),
        CELER_JSON_PAIR(oo, buffer_capacity),
        CELER_JSON_PAIR(oo, auto_flush),
        CELER_JSON_PAIR(oo, max_steps),
        CELER_JSON_PAIR(oo, cherenkov),
        CELER_JSON_PAIR(oo, scintillation),
    };
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

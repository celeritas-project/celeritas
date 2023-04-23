//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/RunnerInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RunnerInputIO.json.hh"

#include <string>

#include "corecel/cont/ArrayIO.json.hh"
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
NLOHMANN_JSON_SERIALIZE_ENUM(
    TrackOrder,
    {{TrackOrder::unsorted, "unsorted"},
     {TrackOrder::shuffled, "shuffled"},
     {TrackOrder::partition_status, "partition-status"},
     {TrackOrder::sort_step_limit_action, "action-id"}})
//---------------------------------------------------------------------------//
}  // namespace celeritas

namespace demo_loop
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, EnergyDiagInput const& v)
{
    j = nlohmann::json{{"axis", std::string(1, v.axis)},
                       {"min", v.min},
                       {"max", v.max},
                       {"num_bins", v.num_bins}};
}

void from_json(nlohmann::json const& j, EnergyDiagInput& v)
{
    std::string temp_axis;
    j.at("axis").get_to(temp_axis);
    CELER_VALIDATE(temp_axis.size() == 1,
                   << "axis spec has length " << temp_axis.size()
                   << " (must be a single character)");
    v.axis = temp_axis.front();
    j.at("min").get_to(v.min);
    j.at("max").get_to(v.max);
    j.at("num_bins").get_to(v.num_bins);
}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, RunnerInput& v)
{
#define LDIO_LOAD_OPTION(NAME)          \
    do                                  \
    {                                   \
        if (j.contains(#NAME))          \
            j.at(#NAME).get_to(v.NAME); \
    } while (0)
#define LDIO_LOAD_REQUIRED(NAME) j.at(#NAME).get_to(v.NAME)

    LDIO_LOAD_OPTION(cuda_heap_size);
    LDIO_LOAD_OPTION(cuda_stack_size);
    LDIO_LOAD_OPTION(environ);

    LDIO_LOAD_REQUIRED(geometry_filename);
    LDIO_LOAD_REQUIRED(physics_filename);
    LDIO_LOAD_OPTION(hepmc3_filename);
    LDIO_LOAD_OPTION(mctruth_filename);

    LDIO_LOAD_OPTION(mctruth_filter);
    LDIO_LOAD_OPTION(primary_gen_options);

    LDIO_LOAD_OPTION(seed);
    LDIO_LOAD_OPTION(max_num_tracks);
    LDIO_LOAD_OPTION(max_steps);
    LDIO_LOAD_REQUIRED(initializer_capacity);
    LDIO_LOAD_REQUIRED(max_events);
    LDIO_LOAD_REQUIRED(secondary_stack_factor);
    LDIO_LOAD_REQUIRED(enable_diagnostics);
    LDIO_LOAD_REQUIRED(use_device);
    LDIO_LOAD_OPTION(sync);

    LDIO_LOAD_OPTION(mag_field);
    LDIO_LOAD_OPTION(field_options);

    LDIO_LOAD_OPTION(step_limiter);
    LDIO_LOAD_OPTION(brem_combined);
    LDIO_LOAD_OPTION(energy_diag);
    LDIO_LOAD_OPTION(track_order);
    LDIO_LOAD_OPTION(geant_options);

#undef LDIO_LOAD_OPTION
#undef LDIO_LOAD_REQUIRED

    CELER_VALIDATE(v.hepmc3_filename.empty() != !v.primary_gen_options,
                   << "either a HepMC3 filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(!v.mctruth_filter || !v.mctruth_filename.empty(),
                   << "'mctruth_filter' cannot be specified without providing "
                      "'mctruth_filename'");
    CELER_VALIDATE(v.mag_field == RunnerInput::no_field()
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
    LDIO_SAVE_OPTION(hepmc3_filename);
    LDIO_SAVE_OPTION(mctruth_filename);

    if (!v.mctruth_filename.empty())
    {
        LDIO_SAVE_REQUIRED(mctruth_filter);
    }
    if (v.hepmc3_filename.empty())
    {
        LDIO_SAVE_REQUIRED(primary_gen_options);
    }

    LDIO_SAVE_OPTION(seed);
    LDIO_SAVE_OPTION(max_num_tracks);
    LDIO_SAVE_OPTION(max_steps);
    LDIO_SAVE_REQUIRED(initializer_capacity);
    LDIO_SAVE_REQUIRED(max_events);
    LDIO_SAVE_REQUIRED(secondary_stack_factor);
    LDIO_SAVE_REQUIRED(enable_diagnostics);
    LDIO_SAVE_REQUIRED(use_device);
    LDIO_SAVE_OPTION(sync);

    LDIO_SAVE_OPTION(mag_field);
    if (v.mag_field != RunnerInput::no_field())
    {
        LDIO_SAVE_REQUIRED(field_options);
    }

    LDIO_SAVE_OPTION(step_limiter);
    LDIO_SAVE_OPTION(brem_combined);

    if (v.enable_diagnostics)
    {
        LDIO_SAVE_REQUIRED(energy_diag);
    }
    LDIO_SAVE_OPTION(track_order);
    if (celeritas::ends_with(v.physics_filename, ".gdml"))
    {
        LDIO_SAVE_REQUIRED(geant_options);
    }

#undef LDIO_SAVE_OPTION
#undef LDIO_SAVE_REQUIRED
}

//---------------------------------------------------------------------------//
}  // namespace demo_loop

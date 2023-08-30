//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RunInputIO.json.hh"

#include "corecel/cont/ArrayIO.json.hh"
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
void from_json(nlohmann::json const& j, PhysicsListSelection& value)
{
    static auto const from_string
        = StringEnumMapper<PhysicsListSelection>::from_cstring_func(
            to_cstring, "physics list");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, PhysicsListSelection const& value)
{
    j = std::string{to_cstring(value)};
}

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
#define RI_LOAD_REQUIRED(NAME) j.at(#NAME).get_to(v.NAME)

    RI_LOAD_REQUIRED(geometry_file);
    RI_LOAD_OPTION(event_file);

    RI_LOAD_OPTION(primary_options);

    RI_LOAD_OPTION(num_track_slots);
    RI_LOAD_OPTION(max_events);
    RI_LOAD_OPTION(max_steps);
    RI_LOAD_OPTION(initializer_capacity);
    RI_LOAD_OPTION(secondary_stack_factor);
    RI_LOAD_OPTION(sync);
    RI_LOAD_OPTION(default_stream);

    RI_LOAD_OPTION(physics_list);
    RI_LOAD_OPTION(physics_options);

    RI_LOAD_OPTION(field_type);
    RI_LOAD_OPTION(field_file);
    RI_LOAD_OPTION(field);
    RI_LOAD_OPTION(field_options);

    RI_LOAD_OPTION(output_file);
    RI_LOAD_OPTION(physics_output_file);
    RI_LOAD_OPTION(offload_output_file);
    RI_LOAD_OPTION(macro_file);
    RI_LOAD_OPTION(root_buffer_size);
    RI_LOAD_OPTION(write_sd_hits);
    RI_LOAD_OPTION(strip_gdml_pointers);

    RI_LOAD_OPTION(step_diagnostic);
    RI_LOAD_OPTION(step_diagnostic_bins);

#undef RI_LOAD_OPTION
#undef RI_LOAD_REQUIRED

    CELER_VALIDATE(v.event_file.empty() != !v.primary_options,
                   << "either a HepMC3 filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(v.physics_list == PhysicsListSelection::geant_physics_list
                       || !j.contains("physics_options"),
                   << "'physics_options' can only be specified for "
                      "'geant_physics_list'");
    CELER_VALIDATE((v.field != RunInput::no_field() || v.field_type == "rzmap")
                       || !j.contains("field_options"),
                   << "'field_options' cannot be specified without providing "
                      "'field'");
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

    RI_SAVE_REQUIRED(geometry_file);
    RI_SAVE_OPTION(event_file);

    if (v.event_file.empty())
    {
        RI_SAVE_REQUIRED(primary_options);
    }

    RI_SAVE_OPTION(num_track_slots);
    RI_SAVE_OPTION(max_events);
    RI_SAVE_OPTION(max_steps);
    RI_SAVE_OPTION(initializer_capacity);
    RI_SAVE_OPTION(secondary_stack_factor);
    RI_SAVE_OPTION(cuda_stack_size);
    RI_SAVE_OPTION(cuda_heap_size);
    RI_SAVE_OPTION(sync);
    RI_SAVE_OPTION(default_stream);

    RI_SAVE_OPTION(physics_list);
    if (v.physics_list == PhysicsListSelection::geant_physics_list)
    {
        RI_SAVE_OPTION(physics_options);
    }

    RI_SAVE_OPTION(field_type);
    RI_SAVE_OPTION(field_file);
    RI_SAVE_OPTION(field);
    RI_SAVE_OPTION(field_options);

    RI_SAVE_OPTION(output_file);
    RI_SAVE_OPTION(physics_output_file);
    RI_SAVE_OPTION(offload_output_file);
    RI_SAVE_OPTION(macro_file);
    RI_SAVE_OPTION(root_buffer_size);
    RI_SAVE_OPTION(write_sd_hits);
    RI_SAVE_OPTION(strip_gdml_pointers);

    RI_SAVE_OPTION(step_diagnostic);
    RI_SAVE_OPTION(step_diagnostic_bins);

#undef RI_SAVE_OPTION
#undef RI_SAVE_REQUIRED
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

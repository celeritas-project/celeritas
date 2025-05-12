//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInputIO.json.cc
//---------------------------------------------------------------------------//
#include "RunInputIO.json.hh"

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringEnumMapper.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "celeritas/TypesIO.json.hh"
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
    auto const& s = j.get<std::string>();
    if (CELER_UNLIKELY(s == "geant_physics_list"))
    {
        // DEPRECATED: remove in v1.0
        CELER_LOG(warning) << "Deprecated option value '" << s
                           << "': use 'celer_em' instead";
        value = PhysicsListSelection::celer_em;
    }
    else
    {
        value = from_string(j.get<std::string>());
    }
}

void to_json(nlohmann::json& j, PhysicsListSelection const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
void from_json(nlohmann::json const& j, SensitiveDetectorType& value)
{
    static auto const from_string
        = StringEnumMapper<SensitiveDetectorType>::from_cstring_func(
            to_cstring, "sensitive detector type");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, SensitiveDetectorType const& value)
{
    j = std::string{to_cstring(value)};
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, RunInput& v)
{
#define RI_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, v, NAME)
#define RI_LOAD_REQUIRED(NAME) CELER_JSON_LOAD_REQUIRED(j, v, NAME)
#define RI_LOAD_DEPRECATED(OLD, NEW) CELER_JSON_LOAD_DEPRECATED(j, v, OLD, NEW)
#define RI_LOAD_DEFAULT(NAME, DEFAULT) \
    CELER_JSON_LOAD_DEFAULT(j, v, NAME, DEFAULT)

    // Check version (if available)
    check_format(j, "celer-g4");

    RI_LOAD_OPTION(cuda_heap_size);
    RI_LOAD_OPTION(cuda_stack_size);
    RI_LOAD_OPTION(environ);

    RI_LOAD_REQUIRED(geometry_file);
    RI_LOAD_OPTION(event_file);

    RI_LOAD_OPTION(primary_options);

    // DEPRECATED: remove in v1.0
    RI_LOAD_DEPRECATED(sync, action_times);

    RI_LOAD_OPTION(num_track_slots);
    RI_LOAD_DEFAULT(num_track_slots,
                    celeritas::Device::num_devices() ? 262144 : 1024);
    RI_LOAD_OPTION(max_steps);
    RI_LOAD_DEFAULT(initializer_capacity, 8 * v.num_track_slots);
    RI_LOAD_OPTION(secondary_stack_factor);
    RI_LOAD_OPTION(action_times);
    if (j.count("default_stream"))
    {
        // DEPRECATED: remove in v1.0
        CELER_LOG(warning) << "Ignoring removed option 'default_stream'";
    }
    RI_LOAD_DEFAULT(auto_flush, v.num_track_slots);

    RI_LOAD_OPTION(track_order);

    RI_LOAD_OPTION(physics_list);
    RI_LOAD_OPTION(physics_options);
    RI_LOAD_OPTION(interpolation);
    RI_LOAD_OPTION(poly_spline_order);

    RI_LOAD_OPTION(field_type);
    RI_LOAD_OPTION(field_file);
    RI_LOAD_OPTION(field);
    RI_LOAD_OPTION(field_options);

    if (auto iter = j.find("enable_sd"); iter != j.end())
    {
        // DEPRECATED: remove in v1.0
        CELER_LOG(warning)
            << R"(Deprecated option 'enable_sd': refactor as 'sd_type')";
        if (iter->get<bool>())
        {
            v.sd_type = SensitiveDetectorType::event_hit;
        }
        else
        {
            v.sd_type = SensitiveDetectorType::none;
        }
    }
    RI_LOAD_OPTION(sd_type);

    RI_LOAD_OPTION(output_file);
    RI_LOAD_OPTION(physics_output_file);
    RI_LOAD_OPTION(offload_output_file);
    RI_LOAD_OPTION(macro_file);

    if (auto iter = j.find("write_sd_hits"); iter != j.end())
    {
        // DEPRECATED: remove in v1.0
        CELER_LOG(warning) << "Deprecated option 'write_sd_hits': disable "
                              "output using CELER_DISABLE_ROOT";
        if (!iter->get<bool>())
        {
            celeritas::environment().insert({"CELER_DISABLE_ROOT", "1"});
        }
    }

    RI_LOAD_OPTION(step_diagnostic);
    RI_LOAD_OPTION(step_diagnostic_bins);
    RI_LOAD_OPTION(slot_diagnostic_prefix);

#undef RI_LOAD_OPTION
#undef RI_LOAD_REQUIRED

    // TODO: move these validation checks to GlobalSetup
    CELER_VALIDATE(v.event_file.empty() == static_cast<bool>(v.primary_options),
                   << "either a HepMC3 filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(v.physics_list != PhysicsListSelection::ftfp_bert
                       || !j.contains("physics_options"),
                   << "'physics_options' can only be specified for "
                      "'celer_ftfp_bert' or 'celer_em'");
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
#define RI_SAVE_OPTION(NAME) \
    CELER_JSON_SAVE_WHEN(j, v, NAME, v.NAME != default_args.NAME)
#define RI_SAVE(NAME) CELER_JSON_SAVE(j, v, NAME)

    j = nlohmann::json::object();
    RunInput const default_args;

    // Save version and format type
    save_format(j, "celer-g4");

    RI_SAVE_OPTION(cuda_stack_size);
    RI_SAVE_OPTION(cuda_heap_size);
    RI_SAVE(environ);

    RI_SAVE(geometry_file);
    RI_SAVE_OPTION(event_file);

    if (v.event_file.empty())
    {
        RI_SAVE(primary_options);
    }

    RI_SAVE(num_track_slots);
    RI_SAVE_OPTION(max_steps);
    RI_SAVE(initializer_capacity);
    RI_SAVE(secondary_stack_factor);
    RI_SAVE(action_times);
    RI_SAVE(auto_flush);

    RI_SAVE(track_order);

    RI_SAVE(physics_list);
    if (v.physics_list != PhysicsListSelection::ftfp_bert)
    {
        RI_SAVE(physics_options);
    }
    RI_SAVE(interpolation);
    RI_SAVE(poly_spline_order);

    RI_SAVE(field_type);
    if (v.field_type == "rzmap")
    {
        RI_SAVE(field_file);
        RI_SAVE(field_options);
    }
    else if (v.field != RunInput::no_field())
    {
        RI_SAVE(field);
        RI_SAVE(field_options);
    }

    RI_SAVE(sd_type);

    RI_SAVE(output_file);
    RI_SAVE(physics_output_file);
    RI_SAVE(offload_output_file);
    RI_SAVE(macro_file);

    RI_SAVE(step_diagnostic);
    RI_SAVE_OPTION(step_diagnostic_bins);
    RI_SAVE_OPTION(slot_diagnostic_prefix);

#undef RI_SAVE_OPTION
#undef RI_SAVE
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

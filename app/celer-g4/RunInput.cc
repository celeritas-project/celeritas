//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.cc
//---------------------------------------------------------------------------//
#include "RunInput.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "accel/SharedParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list selection.
 */
char const* to_cstring(PhysicsListSelection value)
{
    static EnumStringMapper<PhysicsListSelection> const to_cstring_impl{
        "ftfp_bert",
        "celer_ftfp_bert",
        "celer_em",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the physics list selection.
 */
char const* to_cstring(SensitiveDetectorType value)
{
    static EnumStringMapper<SensitiveDetectorType> const to_cstring_impl{
        "none",
        "simple_calo",
        "event_hit",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the run arguments are valid.
 */
RunInput::operator bool() const
{
    return !geometry_file.empty() && (primary_options || !event_file.empty())
           && physics_list < PhysicsListSelection::size_
           && (field == no_field() || field_options)
           && ((num_track_slots > 0 && max_steps > 0
                && initializer_capacity > 0 && secondary_stack_factor > 0
                && auto_flush > 0 && auto_flush <= initializer_capacity)
               || SharedParams::CeleritasDisabled())
           && (step_diagnostic_bins > 0 || !step_diagnostic);
}

//---------------------------------------------------------------------------
// Convert RunInput to celeritas::inp::Input
//---------------------------------------------------------------------------

celeritas::inp::Input to_input(const celeritas::app::RunInput& run_input)
{
    using namespace celeritas;

    inp::Input result;

    // Environment options
    if (run_input.cuda_stack_size != RunInput::unspecified ||
        run_input.cuda_heap_size != RunInput::unspecified)
    {
        inp::Device device;
        device.stack_size = run_input.cuda_stack_size;
        device.heap_size = run_input.cuda_heap_size;
        result.tuning.device = std::move(device);
    }

    // Problem definition
    result.geometry_file = run_input.geometry_file;

    if (!run_input.event_file.empty())
    {
        inp::ReadFileEvents events;
        events.event_file = run_input.event_file;
        result.events = std::move(events);
    }
    else if (run_input.primary_options)
    {
        inp::PrimaryGenerator generator;
        generator = run_input.primary_options; // Assuming compatibility
        result.events = std::move(generator);
    }

    // Control options
    {
        inp::StateCapacity capacity;
        capacity.tracks = run_input.num_track_slots;
        capacity.initializers = run_input.initializer_capacity;
        capacity.secondaries = static_cast<size_type>(
            run_input.secondary_stack_factor * run_input.num_track_slots);
        capacity.primaries = run_input.auto_flush;
        result.tuning.capacity = std::move(capacity);
    }

    {
        inp::TrackingLimits limits;
        limits.steps = run_input.max_steps;
        limits.field_substeps = run_input.auto_flush; // Assuming similarity
        result.tracking.limits = std::move(limits);
    }

    // Physics setup
    result.physics.ignore_processes = run_input.physics_options.ignore_processes;

    // Field setup
    if (run_input.field != RunInput::no_field())
    {
        inp::UniformField field;
        field.strength = run_input.field;
        field.driver_options = run_input.field_options;
        result.field = std::move(field);
    }

    // Sensitive detector
    if (run_input.sd_type != RunInput::SensitiveDetectorType::none)
    {
        inp::Scoring scoring;
        if (run_input.sd_type == RunInput::SensitiveDetectorType::simple_calo)
        {
            scoring.simple_calo.emplace();
        }
        else if (run_input.sd_type == RunInput::SensitiveDetectorType::event_hit)
        {
            scoring.sd.emplace(); // Assuming default SensitiveDetector
        }
        result.scoring = std::move(scoring);
    }

    // Diagnostics
    result.diagnostics.output_file = run_input.output_file;
    result.diagnostics.export_files.physics = run_input.physics_output_file;
    result.diagnostics.export_files.offload = run_input.offload_output_file;
    result.diagnostics.timers.action = run_input.action_times;

    if (!run_input.slot_diagnostic_prefix.empty())
    {
        inp::SlotDiagnostic slot_diag;
        slot_diag.basename = run_input.slot_diagnostic_prefix;
        result.diagnostics.slot = std::move(slot_diag);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

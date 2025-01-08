//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.cc
//---------------------------------------------------------------------------//
#include "RunInput.hh"

#include <fstream>

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/inp/Input.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
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

//---------------------------------------------------------------------------//
inp::Input to_input(RunInput const& run_input)
{
    using namespace celeritas;

    inp::Input result;

    result.tuning.environ
        = {run_input.environ.begin(), run_input.environ.end()};

    // TODO: add option to enable/disable rather than checking device/env
    if (celeritas::device())
    {
        inp::Device device;
        device.stack_size = run_input.cuda_stack_size;
        device.heap_size = run_input.cuda_heap_size;
        device.default_stream = run_input.default_stream;
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
        result.events = to_input(run_input.primary_options);
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
        result.tracking.limits = std::move(limits);
    }

    result.tuning.track_order = [&] {
        auto track_order = run_input.track_order;
        if (track_order != TrackOrder::size_)
            return track_order;

        if (result.tuning.device)
        {
            // Device is activated: initializing by charge is more performant
            return TrackOrder::init_charge;
        }

        // Device is not active: don't sort
        return TrackOrder::none;
    }();

    // Field setup
    if (run_input.field_type == "rzmap")
    {
        CELER_LOG_LOCAL(info)
            << "Loading RZMapField from " << run_input.field_file;
        std::ifstream inp(run_input.field_file);
        CELER_VALIDATE(inp,
                       << "failed to open field map file at '"
                       << run_input.field_file << "'");

        // Read RZ map from file
        RZMapFieldInput rzmap;
        inp >> rzmap;

        // Replace driver options with user options
        rzmap.driver_options = run_input.field_options;

        result.field = std::move(rzmap);
    }
    else if (run_input.field_type == "uniform")
    {
        inp::UniformField field;
        field.strength = run_input.field;

        auto field_val = norm(field.strength);
        CELER_LOG_LOCAL(info)
            << "Using a uniform field " << field_val << " [T]";
        if (field_val > 0)
        {
            field.driver_options = run_input.field_options;
            result.field = std::move(field);
        }
    }
    else
    {
        CELER_VALIDATE(
            false, << "invalid field type '" << run_input.field_type << "'");
    }

    if (run_input.sd_type != SensitiveDetectorType::none)
    {
        // Activate Geant4 SD callbacks
        result.scoring.sd.emplace();
    }

    // Diagnostics
    auto& d = result.diagnostics;
    d.output_file = run_input.output_file;
    d.export_files.physics = run_input.physics_output_file;
    d.export_files.offload = run_input.offload_output_file;
    d.timers.action = run_input.action_times;

    if (!run_input.slot_diagnostic_prefix.empty())
    {
        inp::SlotDiagnostic slot_diag;
        slot_diag.basename = run_input.slot_diagnostic_prefix;
        d.slot = std::move(slot_diag);
    }

    if (run_input.step_diagnostic)
    {
        inp::StepDiagnostic step;
        step.bins = run_input.step_diagnostic_bins;
        d.step = std::move(step);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

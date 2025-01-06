//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.cc
//---------------------------------------------------------------------------//
#include "RunnerInput.hh"

#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/inp/Input.hh"

#include "PrimaryGeneratorOptions.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
inp::Input to_input(RunnerInput const& r)
{
    inp::Input input;

    // Geometry and event configurations
    input.geometry_file = r.geometry_file;

    if (!r.event_file.empty())
    {
        if (r.file_sampling_options)
        {
            inp::SampleFileEvents sfe;
            sfe.num_events = r.file_sampling_options.num_events;
            sfe.num_merged = r.file_sampling_options.num_merged;
            sfe.event_file = r.event_file;
            input.events = std::move(sfe);
        }
        else
        {
            inp::ReadFileEvents rfe;
            rfe.event_file = r.event_file;
            input.events = std::move(rfe);
        }
    }
    else if (r.primary_options)
    {
        input.events = to_input(r.primary_options);  // Existing conversion
                                                     // logic
    }
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }

    // Magnetic field
    if (r.field != RunnerInput::no_field())
    {
        inp::UniformField field;
        field.strength = r.field;
        field.driver_options = r.field_options;
        input.field = field;
    }
    else
    {
        input.field = inp::NoField{};
    }

    // Diagnostics
    if (!r.mctruth_file.empty())
    {
        inp::McTruth mct;
        mct.output_file = r.mctruth_file;
        mct.filter = r.mctruth_filter;
        input.diagnostics.mctruth = std::move(mct);
    }
    input.diagnostics.perfetto_file = r.tracing_file;
    input.diagnostics.timers.action = r.action_times;
    input.diagnostics.timers.step = r.write_step_times;
    input.diagnostics.action = r.action_diagnostic;
    if (!r.slot_diagnostic_prefix.empty())
    {
        inp::SlotDiagnostic slot_diag;
        slot_diag.basename = r.slot_diagnostic_prefix;
        input.diagnostics.slot = std::move(slot_diag);
    }
    if (r.step_diagnostic)
    {
        inp::StepDiagnostic step_diag;
        step_diag.bins = r.step_diagnostic_bins;
        input.diagnostics.step = std::move(step_diag);
    }
    input.diagnostics.step_counters = r.write_track_counts;

    // Tuning
    inp::StateCapacity capacity;
    capacity.tracks = r.num_track_slots;
    capacity.initializers = r.initializer_capacity;
    capacity.secondaries = r.secondary_stack_factor * r.num_track_slots;
    input.tuning.capacity = capacity;

    if (r.use_device)
    {
        inp::Device device;
        device.stack_size = r.cuda_stack_size;
        device.heap_size = r.cuda_heap_size;
        device.default_stream = r.default_stream;
        input.tuning.device = std::move(device);
    }

    input.tuning.warm_up = r.warm_up;

    // Environment
    input.tuning.environ = {r.environ.begin(), r.environ.end()};

    // Physics
    inp::EmPhysicsOptions em_options;
    em_options.brem_combined = r.brem_combined;

    // Spline energy loss order
    if (r.spline_eloss_order == 2)
    {
        em_options.eloss_spline = true;
    }
    else if (r.spline_eloss_order == 1)
    {
        em_options.eloss_spline = false;
    }
    else
    {
        CELERITAS_VALIDATE(
            false, << "Invalid spline_eloss_order: " << r.spline_eloss_order);
    }

    // Step limiter for charged particles
    em_options.step_limit = r.step_limiter;

    input.physics.em_options = em_options;

    // Tracking
    inp::TrackingLimits tracking_limits;
    tracking_limits.steps = r.max_steps;
    input.tracking.limits = tracking_limits;

    // Optical options
    if (r.optical)
    {
        inp::StateCapacity optical_capacity;
        optical_capacity.tracks = r.optical.num_track_slots;
        optical_capacity.initializers = r.optical.initializer_capacity;
        optical_capacity.primaries = r.optical.auto_flush;
        input.tuning.optical_capacity = std::move(optical_capacity);
    }

    // Simple calorimeter scoring
    if (!r.simple_calo.empty())
    {
        inp::SimpleCalo calo;
        calo.volumes = r.simple_calo;
        input.scoring.simple_calo = std::move(calo);
    }

    // Optional fields not in RunnerInput
    // - `physics_file`: No equivalent field in the new API.
    // - `merge_events`: No equivalent field in the new API.

    return input;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

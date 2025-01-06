//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.cc
//---------------------------------------------------------------------------//
#include "RunnerInput.hh"

#include <limits>

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
    inp::Input result;

    // Geometry and event configurations
    result.geometry_file = r.geometry_file;

    if (!run_input.event_file.empty())
    {
        if (r.file_sampling_options)
        {
            inp::SampleFileEvents sfe;
            sfe.num_events = r.file_sampling_options.num_events;
            sfe.num_merged = r.file_sampling_options.num_merged;
            sfe.event_file = r.event_file;
            result.events = std::move(sfe);
        }
        else
        {
            inp::ReadFileEvents rfe;
            rfe.event_file = r.event_file;
            result.events = std::move(rfe);
        }
    }
    else if (run_input.primary_options)
    {
        inp::PrimaryGenerator generator;
        result.events = to_input(r.primary_options);
    }

    // Magnetic field
    if (r.field == RunnerInput::no_field())
    {
        result.field = inp::NoField{};
    }
    else
    {
        inp::UniformField field;
        field.strength = r.field;
        field.driver_options = r.field_options;
        result.field = field;
    }

    // Diagnostics
    auto& d = result.diagnostics;
    if (!r.mctruth_file.empty())
    {
        inp::McTruth mct;
        mct.output_file = r.mctruth_file;
        mct.filter = r.mctruth_filter;
        d.mctruth = std::move(mct);
    }
    d.perfetto_file = r.tracing_file;
    d.timers.action = r.action_times;
    d.timers.step = r.write_step_times;
    d.action = r.action_diagnostic;
    if (!r.slot_diagnostic_prefix.empty())
    {
        inp::SlotDiagnostic slot_diag;
        slot_diag.basename = r.slot_diagnostic_prefix;
        d.slot = std::move(slot_diag);
    }
    if (r.step_diagnostic)
    {
        inp::StepDiagnostic step_diag;
        step_diag.bins = r.step_diagnostic_bins;
        d.step = std::move(step_diag);
    }
    d.step_counters = r.write_track_counts;

    // Tuning
    {
        inp::StateCapacity capacity;
        capacity.tracks = r.num_track_slots;
        capacity.initializers = r.initializer_capacity;
        capacity.secondaries = r.secondary_stack_factor * r.num_track_slots;

        // TODO: replace "max" with # events during construction?
        constexpr auto LimitsT
            = numeric_limits<decltype(capacity.events)> capacity.events
            = r.merge_events ? LimitsT::max() : 0;

        result.tuning.capacity = capacity;
    }

    if (r.use_device)
    {
        inp::Device device;
        device.stack_size = r.cuda_stack_size;
        device.heap_size = r.cuda_heap_size;
        device.default_stream = r.default_stream;
        result.tuning.device = std::move(device);
    }

    result.tuning.warm_up = r.warm_up;

    // Environment
    result.tuning.environ = {r.environ.begin(), r.environ.end()};

    // Physics
    {
        inp::EmPhysicsOptions em_options;
        em_options.brem_combined = r.brem_combined;

        // Spline energy loss order
        CELER_VALIDATE(r.spline_eloss_order > 0 && r.spline_eloss_order <= 2,
                       "unsupported energy loss spline order "
                           << r.spline_eloss_order);
        em_options.eloss_spline = (r.spline_eloss_order == 2);

        // Step limiter for charged particles
        em_options.step_limit = r.step_limiter;

        result.physics.em_options = std::move(em_options);
        result.physics.physics_file = r.physics_file;
    }

    // Tracking
    inp::TrackingLimits tracking_limits;
    tracking_limits.steps = r.max_steps;
    result.tracking.limits = tracking_limits;

    // Optical options
    if (r.optical)
    {
        inp::StateCapacity optical_capacity;
        optical_capacity.tracks = r.optical.num_track_slots;
        optical_capacity.initializers = r.optical.initializer_capacity;
        optical_capacity.primaries = r.optical.auto_flush;
        result.tuning.optical_capacity = std::move(optical_capacity);
    }

    // Simple calorimeter scoring
    if (!r.simple_calo.empty())
    {
        inp::SimpleCalo calo;
        calo.volumes = r.simple_calo;
        result.scoring.simple_calo = std::move(calo);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.cc
//---------------------------------------------------------------------------//
#include "RunnerInput.hh"

#include <limits>

#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/inp/Import.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
inp::System load_system(RunnerInput const& ri)
{
    inp::System s;

    s.environment = {ri.environ.begin(), ri.environ.end()};

    if (ri.use_device)
    {
        s.device = [&ri] {
            inp::Device d;
            d.stack_size = ri.cuda_stack_size;
            d.heap_size = ri.cuda_heap_size;
            return d;
        }();
    }
    return s;
}

//---------------------------------------------------------------------------//
inp::Problem load_problem(RunnerInput const& ri)
{
    inp::Problem p;

    // Geometry and event configurations
    p.model.geometry = ri.geometry_file;

    // Magnetic field
    if (ri.field == RunnerInput::no_field())
    {
        p.field = inp::NoField{};
    }
    else
    {
        inp::UniformField field;
        field.strength = ri.field;
        field.driver_options = ri.field_options;
        p.field = field;
    }

    // Diagnostics
    {
        auto& d = p.diagnostics;
        if (!ri.mctruth_file.empty())
        {
            d.mctruth = [&ri] {
                inp::McTruth mct;
                mct.output_file = ri.mctruth_file;
                mct.filter = ri.mctruth_filter;
                return mct;
            }();
        }
        d.perfetto_file = ri.tracing_file;
        d.timers.action = ri.action_times;
        d.timers.step = ri.write_step_times;
        d.action = ri.action_diagnostic;
        if (!ri.slot_diagnostic_prefix.empty())
        {
            d.slot = inp::SlotDiagnostic{ri.slot_diagnostic_prefix};
        }
        if (ri.step_diagnostic)
        {
            d.step = [&ri] {
                inp::StepDiagnostic step_diag;
                step_diag.bins = ri.step_diagnostic_bins;
                return step_diag;
            }();
        }
        d.counters.step = ri.write_track_counts;
        d.counters.event = ri.transporter_result;
    }

    // Control
    {
        inp::StateCapacity capacity;
        capacity.tracks = ri.num_track_slots;
        capacity.initializers = ri.initializer_capacity;
        capacity.secondaries = static_cast<size_type>(ri.secondary_stack_factor
                                                      * ri.num_track_slots);

        // TODO: replace "max" with # events during construction?
        using LimitsT = std::numeric_limits<decltype(capacity.events)>;
        capacity.events = ri.merge_events ? LimitsT::max() : 0;

        p.control.capacity = capacity;

        p.control.warm_up = ri.warm_up;
        p.control.seed = ri.seed;

        // TODO: set number of streams
        p.control.num_streams = 1;

        if (ri.use_device)
        {
            p.control.device_debug = [&ri] {
                inp::DeviceDebug dd;
                dd.default_stream = ri.default_stream;
                dd.sync_stream = ri.action_times;
                return dd;
            }();
        }
        p.control.track_order = ri.track_order;
    }

    // Physics
    {
        CELER_ASSERT(p.physics.em);
        auto& em = *p.physics.em;

        CELER_ASSERT(em.brems);
        em.brems->combined_model = ri.brem_combined;

        // Spline energy loss order
        CELER_VALIDATE(ri.spline_eloss_order == 1 || ri.spline_eloss_order == 3,
                       << "unsupported energy loss spline order "
                       << ri.spline_eloss_order);
        em.eloss_spline = (ri.spline_eloss_order == 3);
    }

    // Tracking
    p.tracking.limits.steps = ri.max_steps;
    p.tracking.force_step_limit = ri.step_limiter;

    // Optical options
    if (ri.optical)
    {
        p.control.optical_capacity = [&ri] {
            inp::StateCapacity sc;
            sc.tracks = ri.optical.num_track_slots;
            sc.initializers = ri.optical.initializer_capacity;
            sc.primaries = ri.optical.auto_flush;
            return sc;
        }();
    }

    // Simple calorimeter scoring
    if (!ri.simple_calo.empty())
    {
        p.scoring.simple_calo = inp::SimpleCalo{ri.simple_calo};
    }

    return p;
}

//---------------------------------------------------------------------------//
inp::Events load_events(RunnerInput const& ri)
{
    CELER_VALIDATE(ri.event_file.empty() != !ri.primary_options,
                   << "either a event filename or options to generate "
                      "primaries must be provided (but not both)");

    if (!ri.event_file.empty())
    {
        if (ri.file_sampling_options)
        {
            inp::SampleFileEvents sfe;
            sfe.num_events = ri.file_sampling_options.num_events;
            sfe.num_merged = ri.file_sampling_options.num_merged;
            sfe.event_file = ri.event_file;
            sfe.seed = ri.seed;
            return sfe;
        }

        inp::ReadFileEvents rfe;
        rfe.event_file = ri.event_file;
        return rfe;
    }

    CELER_ASSERT(ri.primary_options);
    return to_input(ri.primary_options);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Convert to standalone input format.
 */
inp::StandaloneInput to_input(RunnerInput const& ri)
{
    inp::StandaloneInput si;

    si.system = load_system(ri);
    si.problem = load_problem(ri);
    if (!ri.physics_file.empty())
    {
        // Read ROOT input
        si.physics_import = inp::FileImport{ri.physics_file};
    }
    else
    {
        // Set up Geant4
        si.geant_setup = ri.physics_options;
        si.physics_import = inp::GeantImport{};
    }
    si.geant_data = inp::GeantDataImport{};
    si.events = load_events(ri);

    return si;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

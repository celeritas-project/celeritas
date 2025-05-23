//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.cc
//---------------------------------------------------------------------------//
#include "RunnerInput.hh"

#include <limits>
#include <map>
#include <optional>
#include <variant>

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/StringUtils.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/inp/Diagnostics.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/inp/Field.hh"
#include "celeritas/inp/Model.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/inp/PhysicsProcess.hh"
#include "celeritas/inp/Scoring.hh"
#include "celeritas/inp/System.hh"
#include "celeritas/inp/Tracking.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/io/RootEventReader.hh"
#ifdef _OPENMP
#    include <omp.h>
#endif

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
/*!
 * Get the number of streams from the number of OpenMP threads.
 *
 * The OMP_NUM_THREADS environment variable can be used to control the number
 * of threads/streams. The value of OMP_NUM_THREADS should be a list of
 * positive integers, each of which sets the number of threads for the parallel
 * region at the corresponding nested level. The number of streams is set to
 * the first value in the list. If OMP_NUM_THREADS is not set, the value will
 * be implementation defined.
 */
size_type get_num_streams(bool merge_events)
{
    size_type result = 1;
#if CELERITAS_OPENMP == CELERITAS_OPENMP_EVENT
    if (!merge_events)
    {
#    pragma omp parallel
        {
            if (omp_get_thread_num() == 0)
            {
                result = omp_get_num_threads();
            }
        }
    }
#else
    CELER_DISCARD(merge_events);
#endif

    // TODO: Don't create more streams than events
    return result;
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
        d.status_checker = ri.status_checker;
    }

    // Control
    {
        // NOTE: old celer-sim input is *integrated* over streams
        inp::CoreStateCapacity capacity;
        capacity.primaries = 0;  // Immediately generate initializers
        capacity.initializers = ri.initializer_capacity;
        capacity.tracks = ri.num_track_slots;
        capacity.secondaries = static_cast<size_type>(ri.secondary_stack_factor
                                                      * ri.num_track_slots);

        // TODO: replace "max" with # events during construction
        if (ri.merge_events)
        {
            using LimitsT = std::numeric_limits<decltype(capacity.events)>;
            capacity.events = LimitsT::max();
        }

        p.control.capacity = capacity;

        p.control.warm_up = ri.warm_up;
        p.control.seed = ri.seed;
        p.control.num_streams = get_num_streams(ri.merge_events);

        if (ri.use_device)
        {
            p.control.device_debug = [&ri] {
                inp::DeviceDebug dd;
                dd.sync_stream = ri.action_times;
                return dd;
            }();
        }
        p.control.track_order = ri.track_order;
    }

    // Tracking
    p.tracking.limits.steps = ri.max_steps;
    p.tracking.force_step_limit = ri.step_limiter;
    if (!std::holds_alternative<inp::NoField>(p.field))
    {
        p.tracking.limits.field_substeps = ri.field_options.max_substeps;
    }

    // Optical options
    if (ri.optical)
    {
        p.control.optical_capacity = [&ri] {
            inp::OpticalStateCapacity sc;
            sc.primaries = ri.optical.auto_flush;
            sc.initializers = ri.optical.initializer_capacity;
            sc.tracks = ri.optical.num_track_slots;
            sc.generators = ri.optical.buffer_capacity;
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
        CELER_VALIDATE(
            ends_with(ri.physics_file, ".root"),
            << R"(physics_file must be a ROOT input: use GDML for geometry_file and if forcing an ORANGE geometry, use the `ORANGE_FORCE_INPUT` environment variable)");
        // Read ROOT input
        si.physics_import = inp::FileImport{ri.physics_file};
    }
    else
    {
        // Set up Geant4
        si.geant_setup = ri.physics_options;

        inp::GeantImport geant_import;
        CELER_VALIDATE(
            ri.poly_spline_order == 1
                || ri.interpolation == InterpolationType::poly_spline,
            << "piecewise polynomial spline order cannot be set if "
               "linear or cubic spline interpolation is enabled");
        geant_import.data_selection.interpolation.type = ri.interpolation;
        geant_import.data_selection.interpolation.order = ri.poly_spline_order;
        si.physics_import = std::move(geant_import);
    }
    si.events = load_events(ri);

    // Load actual number of events, needed to contruct core state before
    // loading events
    auto num_events = std::visit(
        Overload{
            [](inp::PrimaryGenerator const& pg) { return pg.num_events; },
            [](inp::SampleFileEvents const& sfe) { return sfe.num_events; },
            [](inp::ReadFileEvents const& rfe) {
                if (ends_with(rfe.event_file, ".root"))
                {
                    return RootEventReader{rfe.event_file, nullptr}.num_events();
                }

                return EventReader{rfe.event_file, nullptr}.num_events();
            },
        },
        si.events);
    CELER_ASSERT(num_events > 0);

    // Save number of events
    auto& ctl = std::get<inp::Problem>(si.problem).control;
    ctl.capacity.events = num_events;
    // Limit number of streams
    ctl.num_streams = std::min(ctl.num_streams, num_events);

    return si;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

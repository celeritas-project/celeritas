//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.cc
//---------------------------------------------------------------------------//
#include "RunInput.hh"

#include <fstream>

#include "corecel/Config.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "accel/SharedParams.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
inp::System load_system(RunInput const& ri)
{
    inp::System s;

    if (celeritas::Device::num_devices())
    {
        inp::Device d;
        d.stack_size = ri.cuda_stack_size;
        d.heap_size = ri.cuda_heap_size;
        s.device = std::move(d);
    }

    s.environment = {ri.environ.begin(), ri.environ.end()};

    return s;
}

//---------------------------------------------------------------------------//
inp::Problem load_problem(RunInput const& ri)
{
    inp::Problem p;

    // Model definition
    p.model.geometry = ri.geometry_file;

    p.control.num_streams = get_geant_num_threads();

    // NOTE: old SetupOptions input *per stream*, but inp::Problem needs
    // *integrated* over streams
    p.control.capacity = [&ri, num_streams = p.control.num_streams] {
        inp::CoreStateCapacity c;
        c.tracks = ri.num_track_slots * num_streams;
        c.initializers = ri.initializer_capacity * num_streams;
        c.secondaries = static_cast<size_type>(
            ri.secondary_stack_factor * ri.num_track_slots * num_streams);
        c.primaries = ri.auto_flush;
        return c;
    }();

    if (celeritas::Device::num_devices())
    {
        inp::DeviceDebug dd;
        dd.sync_stream = ri.action_times;
        p.control.device_debug = std::move(dd);
    }

    if (ri.track_order != TrackOrder::size_)
    {
        p.control.track_order = ri.track_order;
    }

    {
        inp::TrackingLimits& limits = p.tracking.limits;
        limits.steps = ri.max_steps;
    }

    // Field setup
    if (ri.field_type == "rzmap")
    {
        CELER_LOG(info) << "Loading RZMapField from " << ri.field_file;
        std::ifstream inp(ri.field_file);
        CELER_VALIDATE(inp,
                       << "failed to open field map file at '" << ri.field_file
                       << "'");

        // Read RZ map from file
        RZMapFieldInput rzmap;
        inp >> rzmap;

        // Replace driver options with user options
        rzmap.driver_options = ri.field_options;

        p.field = std::move(rzmap);
    }
    else if (ri.field_type == "uniform")
    {
        inp::UniformField field;
        field.strength = ri.field;

        auto field_val = norm(field.strength);
        if (field_val > 0)
        {
            CELER_LOG(info) << "Using a uniform field " << field_val << " [T]";
            field.driver_options = ri.field_options;
            p.field = std::move(field);
        }
    }
    else
    {
        CELER_VALIDATE(false,
                       << "invalid field type '" << ri.field_type << "'");
    }

    if (ri.sd_type != SensitiveDetectorType::none)
    {
        // Activate Geant4 SD callbacks
        p.scoring.sd.emplace();
    }

    {
        // Diagnostics
        auto& d = p.diagnostics;
        d.output_file = ri.output_file;
        d.export_files.physics = ri.physics_output_file;
        d.export_files.offload = ri.offload_output_file;
        d.timers.action = ri.action_times;
        d.perfetto_file = ri.tracing_file;

        if (!ri.slot_diagnostic_prefix.empty())
        {
            inp::SlotDiagnostic slot_diag;
            slot_diag.basename = ri.slot_diagnostic_prefix;
            d.slot = std::move(slot_diag);
        }

        if (ri.step_diagnostic)
        {
            inp::StepDiagnostic step;
            step.bins = ri.step_diagnostic_bins;
            d.step = std::move(step);
        }
    }

    CELER_VALIDATE(ri.macro_file.empty(),
                   << "macro file is no longer supported");

    return p;
}

//---------------------------------------------------------------------------//
inp::Events load_events(RunInput const& ri)
{
    CELER_VALIDATE(ri.event_file.empty() != !ri.primary_options,
                   << "either a event filename or options to generate "
                      "primaries must be provided (but not both)");

    if (!ri.event_file.empty())
    {
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
           && (num_track_slots > 0 && max_steps > 0 && initializer_capacity > 0
               && secondary_stack_factor > 0 && auto_flush > 0
               && auto_flush <= initializer_capacity)
           && (step_diagnostic_bins > 0 || !step_diagnostic);
}

//---------------------------------------------------------------------------//
/*!
 * Convert to standalone input format.
 */
inp::StandaloneInput to_input(RunInput const& ri)
{
    inp::StandaloneInput si;

    si.system = load_system(ri);
    si.problem = load_problem(ri);

    // Set up Geant4
    if (ri.physics_list == PhysicsListSelection::celer_ftfp_bert)
    {
        // TODO: physics list is handled elsewhere
    }
    else
    {
        CELER_VALIDATE(ri.physics_list == PhysicsListSelection::celer_em,
                       << "invalid physics list selection '"
                       << to_cstring(ri.physics_list) << "' (must be 'celer')");
    }

    si.geant_setup = ri.physics_options;

    inp::PhysicsFromGeant geant_import;
    geant_import.ignore_processes.push_back("CoulombScat");
    geant_import.data_selection.interpolation.type = ri.interpolation;
    geant_import.data_selection.interpolation.order = ri.poly_spline_order;
    si.physics_import = std::move(geant_import);

    si.events = load_events(ri);

    return si;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

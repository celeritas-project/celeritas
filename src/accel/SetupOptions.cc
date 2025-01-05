//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include "geocel/GeantGeoUtils.hh"
#include "celeritas/inp/Input.hh"

#include "ExceptionConverter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find volumes by name for SDSetupOptions.
 *
 * Example:
 * \code
   setup.sd.force_volumes = FindVolumes({"foo", "bar"});
 * \endcode
 */
std::unordered_set<G4LogicalVolume const*>
FindVolumes(std::unordered_set<std::string> names)
{
    ExceptionConverter call_g4exception{"celer0006"};
    std::unordered_set<G4LogicalVolume const*> result;
    CELER_TRY_HANDLE(result = find_geant_volumes(std::move(names)),
                     call_g4exception);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert to Celeritas input.
 */
inp::Input to_input(SetupOptions const& so)
{
    inp::Input i;

    i.geometry_file = so.geometry_file;
    i.diagnostics.output_file = so.output_file;

    {
        inp::StateCapacity c;
        c.tracks = so.max_num_tracks;
        c.initializers = so.initializer_capacity;
        c.secondaries = so.secondary_stack_factor * c.tracks;
        c.primaries = so.auto_flush;

        i.tuning.capacity = std::move(c);
    }
    {
        inp::TrackingLimits tl;
        tl.events = so.max_num_events;
        tl.steps = so.max_steps;
        tl.step_iters = so.max_step_iters;
        tl.field_substeps = so.max_field_substeps;

        i.tracking.limits = std::move(tl);
    }

    // TODO: add option to enable/disable rather than checking device/env
    if (celeritas::device())
    {
        inp::Device d;
        d.default_stream = r.default_stream;
        d.stack_size = r.cuda_stack_size;
        d.heap_size = r.cuda_heap_size;

        i.tuning.device = std::move(d);
    }

    i.tuning.track_order = [&] {
        auto track_order = so.track_order;
        if (track_order != TrackOrder::size_)
            return track_order;

        if (i.tuning.device)
        {
            // Device is activated: initializing by charge is more performant
            return TrackOrder::init_charge;
        }

        // Device is not active: don't ort
        return TrackOrder::none;
    }();

    if (so.sd.enabled)
    {
        celeritas::inp::SensitiveDetector sd;

        sd.ignore_zero_deposition = so.sd.ignore_zero_deposition;
        sd.energy_deposition = so.sd.energy_deposition;
        sd.locate_touchable = so.sd.locate_touchable;
        sd.track = so.sd.track;
        sd.pre = so.sd.pre;
        sd.post = so.sd.post;
        sd.force_volumes = so.sd.force_volumes;
        sd.skip_volumes = so.sd.skip_volumes;
        i.scoring.sensitive_detector = std::move(sd);
    }

    i.tuning.num_streams = so.get_num_streams();

    // TODO: map along-step to magnetic field and physics input
    i.physics.make_along_step = so.make_along_step;
    i.physics.ignore_processes = so.ignore_processes;

    {
        ExportFiles ef;
        ef.physics = so.physics_output_file;
        ef.offload = so.offload_output_file;
        ef.geometry = so.geometry_output_file;
        i.diagnostics.ef = std::move(ef);
    }

    i.diagnostics.timers.action = so.action_times;

    if (!so.slot_diagnostic_prefix.empty())
    {
        SlotDiagnostic sd;
        sd.filename_base = slot_diagnostic_prefix;
        i.diagnostics.slot_diagnostic = std::move(sd);
    }

    return i;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

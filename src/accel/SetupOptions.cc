//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include "geocel/GeantGeoUtils.hh"
#include "celeritas/inp/Input.hh"

#include "AlongStepFactory.hh"
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

inp::SDStepPointAttributes to_input(SDSetupOptions::StepPoint const& sp)
{
    inp::SDStepPointAttributes result;
    result.global_time = sp.global_time;
    result.position = sp.position;
    result.direction = sp.direction;
    result.kinetic_energy = sp.kinetic_energy;
    return result;
}

inp::SensitiveDetector to_input(SDSetupOptions const& sd)
{
    celeritas::inp::SensitiveDetector result;

    result.ignore_zero_deposition = sd.ignore_zero_deposition;
    result.energy_deposition = sd.energy_deposition;
    result.locate_touchable = sd.locate_touchable;
    result.track = sd.track;
    result.pre = to_input(sd.pre);
    result.post = to_input(sd.post);
    result.force_volumes = std::set<G4LogicalVolume const*>(
        sd.force_volumes.begin(), sd.force_volumes.end());
    result.skip_volumes = std::set<G4LogicalVolume const*>(
        sd.skip_volumes.begin(), sd.skip_volumes.end());

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
        tl.steps = so.max_steps;
        tl.step_iters = so.max_step_iters;
        tl.field_substeps = so.max_field_substeps;

        i.tracking.limits = std::move(tl);
    }

    // TODO: add option to enable/disable rather than checking device/env
    if (celeritas::device())
    {
        inp::Device d;
        d.default_stream = so.default_stream;
        d.stack_size = so.cuda_stack_size;
        d.heap_size = so.cuda_heap_size;

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
        i.scoring.sd = to_input(so.sd);
    }

    i.tuning.num_streams = so.get_num_streams();

    if (auto* u = so.make_along_step.target<UniformAlongStepFactory>())
    {
        CELER_NOT_IMPLEMENTED("convert uniform factory");
        // Check if magnitude is zero
        i.field = inp::UniformField{};
    }
    else if (auto* u = so.make_along_step.target<RZMapFieldAlongStepFactory>())
    {
        i.field = u->get_field();
    }
    else
    {
        CELER_NOT_IMPLEMENTED("processing generic along-step factory");
    }

    i.physics.ignore_processes = so.ignore_processes;

    {
        inp::ExportFiles ef;
        ef.physics = so.physics_output_file;
        ef.offload = so.offload_output_file;
        ef.geometry = so.geometry_output_file;
        i.diagnostics.export_files = std::move(ef);
    }

    i.diagnostics.timers.action = so.action_times;

    if (!so.slot_diagnostic_prefix.empty())
    {
        inp::SlotDiagnostic sd;
        sd.basename = so.slot_diagnostic_prefix;
        i.diagnostics.slot = std::move(sd);
    }

    return i;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

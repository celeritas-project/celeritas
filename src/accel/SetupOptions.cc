//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGeoUtils.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/inp/FrameworkInput.hh"
#include "celeritas/inp/Problem.hh"

#include "AlongStepFactory.hh"
#include "ExceptionConverter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//

auto to_input(SDSetupOptions::StepPoint const& sp)
{
    inp::GeantSDStepPointAttributes result;
    result.global_time = sp.global_time;
    result.position = sp.position;
    result.direction = sp.direction;
    result.kinetic_energy = sp.kinetic_energy;
    return result;
}

auto to_input(SDSetupOptions const& sd)
{
    inp::GeantSensitiveDetector result;

    result.ignore_zero_deposition = sd.ignore_zero_deposition;
    result.energy_deposition = sd.energy_deposition;
    result.step_length = sd.step_length;
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
 * Construct system attributes from setup options.
 */
inp::System load_system(SetupOptions const& so)
{
    inp::System s;
    if (celeritas::Device::num_devices())
    {
        inp::Device d;
        d.stack_size = so.cuda_stack_size;
        d.heap_size = so.cuda_heap_size;

        s.device = d;
    }
    return s;
}

//---------------------------------------------------------------------------//
/*!
 * FrameworkInput adapter function.
 */
struct ProblemSetup
{
    SetupOptions const& so;

    void operator()(inp::Problem&) const;
};

//---------------------------------------------------------------------------//
/*!
 * Set a Celeritas problem input definition.
 */
void ProblemSetup::operator()(inp::Problem& p) const
{
    if (!so.geometry_file.empty())
    {
        p.model.geometry = so.geometry_file;
    }
    p.diagnostics.output_file = so.output_file;

    p.control.num_streams = so.get_num_streams();

    p.control.capacity = [this, num_streams = p.control.num_streams] {
        inp::CoreStateCapacity c;
        c.tracks = so.max_num_tracks * num_streams;
        c.initializers = so.initializer_capacity * num_streams;
        c.secondaries
            = static_cast<size_type>(so.secondary_stack_factor * c.tracks);
        c.primaries = so.auto_flush;
        return c;
    }();

    p.tracking.limits = [this] {
        inp::TrackingLimits tl;
        tl.steps = so.max_steps;
        tl.step_iters = so.max_step_iters;
        tl.field_substeps = so.max_field_substeps;
        return tl;
    }();

    if (so.track_order != TrackOrder::size_)
    {
        p.control.track_order = so.track_order;
    }

    if (celeritas::Device::num_devices())
    {
        inp::DeviceDebug dd;
        dd.default_stream = so.default_stream;
        dd.sync_stream = so.action_times;

        p.control.device_debug = std::move(dd);
    }

    if (so.sd.enabled)
    {
        p.scoring.sd = to_input(so.sd);
    }

    if (auto* u = so.make_along_step.target<UniformAlongStepFactory>())
    {
        // Check if magnitude is zero
        UniformFieldParams params = u->get_field();
        auto field_val = norm(params.field);
        if (field_val > 0)
        {
            CELER_LOG(info) << "Using a uniform field: " << field_val << " [T]";
            inp::UniformField field;
            field.strength = params.field;
            field.driver_options = params.options;
            p.field = std::move(field);
        }
        else
        {
            CELER_LOG(debug) << "No magnetic field";
        }
    }
    else if (auto* u = so.make_along_step.target<RZMapFieldAlongStepFactory>())
    {
        CELER_LOG(debug) << "Getting RZ map field";
        p.field = u->get_field();
    }
    else
    {
        CELER_NOT_IMPLEMENTED("processing generic along-step factory");
    }

    p.diagnostics.export_files = [this] {
        inp::ExportFiles ef;
        ef.physics = so.physics_output_file;
        ef.offload = so.offload_output_file;
        ef.geometry = so.geometry_output_file;
        return ef;
    }();

    p.diagnostics.timers.action = so.action_times;

    if (!so.slot_diagnostic_prefix.empty())
    {
        p.diagnostics.slot = inp::SlotDiagnostic{so.slot_diagnostic_prefix};
    }
}

//---------------------------------------------------------------------------//
}  // namespace

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
    std::unordered_set<G4LogicalVolume const*> result;
    CELER_TRY_HANDLE(result = find_geant_volumes(std::move(names)),
                     ExceptionConverter{"celer.setup.find_volumes"});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a framework input from setup options.
 *
 * \note The setup options *must* stay in scope until problem initialization!
 */
inp::FrameworkInput to_inp(SetupOptions const& so)
{
    inp::FrameworkInput result;
    result.system = load_system(so);
    result.geant.ignore_processes = so.ignore_processes;
    result.adjuster = ProblemSetup{so};
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

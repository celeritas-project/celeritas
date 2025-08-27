//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SetupOptions.cc
//---------------------------------------------------------------------------//
#include "SetupOptions.hh"

#include <CLHEP/Random/Random.h>
#include <G4ParticleDefinition.hh>

#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/inp/FrameworkInput.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "AlongStepFactory.hh"
#include "ExceptionConverter.hh"

namespace celeritas
{
namespace
{
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
auto to_inp(SDSetupOptions::StepPoint const& sp)
{
    inp::GeantSdStepPointAttributes result;
    result.global_time = sp.global_time;
    result.position = sp.position;
    result.direction = sp.direction;
    result.kinetic_energy = sp.kinetic_energy;
    return result;
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

    p.control.num_streams = [&so = this->so] {
        if (so.get_num_streams)
        {
            return so.get_num_streams();
        }
        return celeritas::get_geant_num_threads();
    }();

    // NOTE: old SetupOptions input *per stream*, but inp::Problem needs
    // *integrated* over streams
    p.control.capacity = [this, num_streams = p.control.num_streams] {
        auto capacity = get_default(so, num_streams);
        inp::CoreStateCapacity c;
        c.tracks = capacity.tracks;
        c.initializers = capacity.initializers;
        c.primaries = capacity.primaries;
        c.secondaries
            = static_cast<size_type>(so.secondary_stack_factor * c.tracks);
        return c;
    }();

    if (so.max_num_events)
    {
        CELER_LOG(warning) << "Ignoring removed option 'max_num_events': will "
                              "be an error in v0.7";
    }

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
        dd.sync_stream = so.action_times;

        p.control.device_debug = std::move(dd);
    }

    p.control.seed = CLHEP::HepRandom::getTheSeed();

    if (so.sd.enabled)
    {
        p.scoring.sd = to_inp(so.sd);
    }

    if (auto* u = so.make_along_step.target<UniformAlongStepFactory>())
    {
        // Check if magnitude is zero
        auto field = u->get_field();
        auto field_val = norm(field.strength);
        if (field_val > 0)
        {
            auto volumes = u->get_volumes();
            auto msg = CELER_LOG(info);
            msg << "Using a uniform field: " << field_val << " [T] in ";
            if (volumes.empty())
            {
                msg << "all";
            }
            else
            {
                msg << volumes.size();
                field.volumes = inp::UniformField::SetVolume{volumes.begin(),
                                                             volumes.end()};
            }
            msg << " volumes";
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
    else if (auto* u = so.make_along_step.target<CylMapFieldAlongStepFactory>())
    {
        CELER_LOG(debug) << "Getting Cyl map field";
        p.field = u->get_field();
    }
    else if (auto* u = so.make_along_step.target<CartMapFieldAlongStepFactory>())
    {
        CELER_LOG(debug) << "Getting covfie cartesian map field";
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

    // Custom user actions
    p.diagnostics.add_user_actions = so.add_user_actions;
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
 * Convert SD options for forward compatibility.
 */
inp::GeantSd to_inp(SDSetupOptions const& sd)
{
    CELER_EXPECT(sd.enabled);
    inp::GeantSd result;

    result.ignore_zero_deposition = sd.ignore_zero_deposition;
    result.energy_deposition = sd.energy_deposition;
    result.step_length = sd.step_length;
    result.track = sd.track;
    result.points[StepPoint::pre] = to_inp(sd.pre);
    result.points[StepPoint::pre].touchable = sd.locate_touchable;
    result.points[StepPoint::post] = to_inp(sd.post);
    result.points[StepPoint::post].touchable = sd.locate_touchable_post;
    result.force_volumes = sd.force_volumes;
    result.skip_volumes = sd.skip_volumes;

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
    using GIDS = GeantImportDataSelection;

    auto includes_muon = [&so]() -> bool {
        return std::any_of(so.offload_particles.begin(),
                           so.offload_particles.end(),
                           [](G4ParticleDefinition* pd) {
                               return (std::abs(pd->GetPDGEncoding())
                                       == pdg::mu_minus().get());
                           });
    };

    inp::FrameworkInput result;
    result.system = load_system(so);
    result.geant.ignore_processes = so.ignore_processes;
    result.geant.data_selection.interpolation = so.interpolation;

    // Correctly assign DataSelection import flags when muons are present
    auto const selection = includes_muon() ? GIDS::em : GIDS::em_basic;
    result.geant.data_selection.particles = selection;
    result.geant.data_selection.processes = selection;

    result.adjust = ProblemSetup{so};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get runtime-dependent default capacity values.
 *
 * \note This must be called after CUDA/MPI have been initialized.
 */
inp::CoreStateCapacity
get_default(SetupOptions const& so, size_type num_streams)
{
    inp::CoreStateCapacity result;
    result.tracks = num_streams * [&so] {
        if (so.max_num_tracks)
        {
            return static_cast<size_type>(so.max_num_tracks);
        }
        if (celeritas::Device::num_devices())
        {
            constexpr size_type device_default = 262144;
            return device_default;
        }
        constexpr size_type host_default = 1024;
        return host_default;
    }();
    result.initializers = so.initializer_capacity
                              ? num_streams * so.initializer_capacity
                              : 8 * result.tracks;
    result.primaries = so.auto_flush ? so.auto_flush
                                     : result.tracks / num_streams;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

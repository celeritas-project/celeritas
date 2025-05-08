//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Problem.cc
//---------------------------------------------------------------------------//
#include "Problem.hh"

#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/params/RngParams.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/alongstep/AlongStepCartMapFieldMscAction.hh"
#include "celeritas/alongstep/AlongStepCylMapFieldMscAction.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/alongstep/AlongStepRZMapFieldMscAction.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSd.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/inp/Diagnostics.hh"
#include "celeritas/inp/Field.hh"
#include "celeritas/inp/Model.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/inp/PhysicsModel.hh"
#include "celeritas/inp/PhysicsProcess.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/inp/Scoring.hh"
#include "celeritas/inp/Tracking.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/RootCoreParamsOutput.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/CherenkovParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/ModelImporter.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/optical/ScintillationParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Process.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/StatusChecker.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/ActionDiagnostic.hh"
#include "celeritas/user/RootStepWriter.hh"
#include "celeritas/user/SimpleCalo.hh"
#include "celeritas/user/SlotDiagnostic.hh"
#include "celeritas/user/StepCollector.hh"
#include "celeritas/user/StepData.hh"
#include "celeritas/user/StepDiagnostic.hh"

namespace celeritas
{
namespace setup
{
namespace
{
//---------------------------------------------------------------------------//
struct GeoBuilder
{
    using result_type = std::shared_ptr<GeoParams>;

    //! Build from filename
    result_type operator()(std::string const& filename)
    {
        CELER_VALIDATE(!filename.empty(),
                       << "empty filename in problem.model.geometry");
        return std::make_shared<GeoParams>(filename);
    }

    //! Build from Geant4
    result_type operator()(G4VPhysicalVolume const* world)
    {
        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // NOTE: this is used to allow a custom "ideal" TestEM3 definition
            // in our regression suite
            static char const fi_hack_envname[] = "ORANGE_FORCE_INPUT";
            auto const& filename = celeritas::getenv(fi_hack_envname);
            if (!filename.empty())
            {
                CELER_LOG(warning)
                    << "Using a temporary, unsupported, and dangerous "
                       "hack to override the ORANGE geometry file: "
                    << fi_hack_envname << "='" << filename << "'";
                return (*this)(filename);
            }
        }
        CELER_VALIDATE(world,
                       << "null world pointer in problem.model.geometry");
        return std::make_shared<GeoParams>(world);
    }
};

auto build_geometry(inp::Model const& m)
{
    return std::visit(GeoBuilder{}, m.geometry);
}

//---------------------------------------------------------------------------//
/*!
 * Construct physics processes.
 */
auto build_physics_processes(inp::EmPhysics const& em,
                             CoreParams::Input const& params,
                             ImportData const& imported)
{
    // TODO: process builder should be deleted; instead it should get
    // p.physics.em or whatever
    std::vector<std::shared_ptr<Process const>> result;
    ProcessBuilder::Options opts;
    if (em.brems)
    {
        opts.brem_combined = em.brems->combined_model;
    }

    ProcessBuilder build_process(
        imported, params.particle, params.material, em.user_processes, opts);
    for (auto pc : ProcessBuilder::get_all_process_classes(imported.processes))
    {
        result.push_back(build_process(pc));
        if (!result.back())
        {
            // Deliberately ignored process
            CELER_LOG(debug) << "Ignored process class " << pc;
            result.pop_back();
        }
    }

    CELER_VALIDATE(!result.empty(),
                   << "no supported physics processes were found");
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct physics.
 */
auto build_physics(inp::Problem const& p,
                   CoreParams::Input const& params,
                   ImportData const& imported)
{
    CELER_ASSUME(p.physics.em);

    PhysicsParams::Input input;
    input.particles = params.particle;
    input.materials = params.material;
    input.action_registry = params.action_reg.get();

    // Set physics options
    input.options.fixed_step_limiter = p.tracking.force_step_limit;
    if (p.control.capacity.secondaries)
    {
        input.options.secondary_stack_factor
            = static_cast<real_type>(*p.control.capacity.secondaries)
              / static_cast<real_type>(p.control.capacity.tracks);
    }
    else
    {
        // Default: twice the number of track slots
        input.options.secondary_stack_factor = 2.0;
    }
    input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
    input.options.disable_integral_xs = !imported.em_params.integral_approach;
    input.options.light.lowest_energy
        = ParticleOptions::Energy(imported.em_params.lowest_electron_energy);
    input.options.heavy.lowest_energy
        = ParticleOptions::Energy(imported.em_params.lowest_muhad_energy);

    // Set multiple scattering options
    input.options.light.range_factor = imported.em_params.msc_range_factor;
    input.options.heavy.range_factor
        = imported.em_params.msc_muhad_range_factor;
    input.options.safety_factor = imported.em_params.msc_safety_factor;
    input.options.lambda_limit = imported.em_params.msc_lambda_limit;
    input.options.light.displaced = imported.em_params.msc_displaced;
    input.options.heavy.displaced = imported.em_params.msc_muhad_displaced;
    input.options.light.step_limit_algorithm
        = imported.em_params.msc_step_algorithm;
    input.options.heavy.step_limit_algorithm
        = imported.em_params.msc_muhad_step_algorithm;

    // Build processes
    input.processes = build_physics_processes(*p.physics.em, params, imported);

    return std::make_shared<PhysicsParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct track initialization params.
 */
auto build_track_init(inp::Control const& c, size_type num_streams)
{
    CELER_VALIDATE(c.capacity.initializers > 0,
                   << "nonpositive capacity.initializers="
                   << c.capacity.initializers);
    CELER_VALIDATE(!c.capacity.events || c.capacity.events > 0,
                   << "nonpositive capacity.events=" << *c.capacity.events);
    // NOTE: if the following assertion fails, a placeholder "event
    // count" should have been changed elsewhere
    CELER_EXPECT(
        !c.capacity.events
        || c.capacity.events
               != std::numeric_limits<decltype(c.capacity.events)>::max());
    TrackInitParams::Input input;
    input.capacity = ceil_div(c.capacity.initializers, num_streams);
    if (c.capacity.events)
    {
        input.max_events = *c.capacity.events;
    }
    else
    {
        // Geant4 integration (TODO: make this a special case)
        input.max_events = 1;
    }
    if (c.track_order)
    {
        input.track_order = *c.track_order;
    }
    else
    {
        if (celeritas::device())
        {
            input.track_order = TrackOrder::init_charge;
        }
        else
        {
            input.track_order = TrackOrder::none;
        }
        CELER_LOG(debug) << "Set default track order " << input.track_order;
    }

    return std::make_shared<TrackInitParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct magnetic field from variant input.
 */
auto build_along_step(inp::Field const& var_field,
                      CoreParams::Input const& params,
                      ImportData const& imported)
{
    bool const eloss = imported.em_params.energy_loss_fluct;
    auto msc = UrbanMscParams::from_import(
        *params.particle, *params.material, imported);

    CELER_ASSUME(!var_field.valueless_by_exception());
    auto next_id = params.action_reg->next_id();
    return std::visit(
        return_as<std::shared_ptr<CoreStepActionInterface>>(Overload{
            [&](inp::NoField const&) {
                using ASA = AlongStepGeneralLinearAction;
                return ASA::from_params(
                    next_id, *params.material, *params.particle, msc, eloss);
            },
            [&](inp::UniformField const& field) {
                using ASA = AlongStepUniformMscAction;
                return ASA::from_params(next_id,
                                        *params.geometry,
                                        *params.material,
                                        *params.particle,
                                        field,
                                        msc,
                                        eloss);
            },
            [&](inp::RZMapField const& field) {
                using ASA = AlongStepRZMapFieldMscAction;
                return ASA::from_params(next_id,
                                        *params.material,
                                        *params.particle,
                                        field,
                                        msc,
                                        eloss);
            },
            [&](inp::CylMapField const& field) {
                using ASA = AlongStepCylMapFieldMscAction;
                return ASA::from_params(next_id,
                                        *params.material,
                                        *params.particle,
                                        field,
                                        msc,
                                        eloss);
            },
            [&](inp::CartMapField const& field) {
                using ASA = AlongStepCartMapFieldMscAction;
                return ASA::from_params(next_id,
                                        *params.material,
                                        *params.particle,
                                        field,
                                        msc,
                                        eloss);
            },
        }),
        var_field);
}

//---------------------------------------------------------------------------//
/*!
 * Construct optical tracking offload.
 */
auto build_optical_offload(inp::OpticalStateCapacity const& cap,
                           CoreParams& params,
                           ImportData const& imported)
{
    using optical::CherenkovParams;
    using optical::MaterialParams;
    using optical::ScintillationParams;

    CELER_VALIDATE(
        !imported.optical_materials.empty(),
        << R"(an optical tracking loop was requested but no optical materials are present)");

    OpticalCollector::Input oc_inp;
    oc_inp.material = MaterialParams::from_import(
        imported, *params.geomaterial(), *params.material());
    oc_inp.cherenkov = std::make_shared<CherenkovParams>(*oc_inp.material);
    oc_inp.scintillation
        = ScintillationParams::from_import(imported, params.particle());

    // Map from optical capacity
    auto num_streams = params.max_streams();
    oc_inp.num_track_slots = ceil_div(cap.tracks, num_streams);
    oc_inp.buffer_capacity = ceil_div(cap.generators, num_streams);
    oc_inp.initializer_capacity = ceil_div(cap.initializers, num_streams);
    oc_inp.auto_flush = ceil_div(cap.primaries, num_streams);

    // Import models
    optical::ModelImporter importer{
        imported, oc_inp.material, params.material()};
    for (auto const& model : imported.optical_models)
    {
        if (auto builder = importer(model.model_class))
        {
            oc_inp.model_builders.push_back(*builder);
        }
    }

    CELER_ASSERT(oc_inp);

    return std::make_shared<OpticalCollector>(params, std::move(oc_inp));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Create "core params" from a problem definition and import data.
 *
 * Conceivably we could rename "core params" someday.
 *
 * \todo Consolidate import data into the problem definition.
 * \todo Migrate the class "Input"/"Option" code into the class itself, using
 * the \c inp namespace definition.
 */
ProblemLoaded problem(inp::Problem const& p, ImportData const& imported)
{
    CELER_LOG(status) << "Initializing problem";

    ScopedMem record_mem("setup::problem");
    ScopedProfiling profile_this{"setup::problem"};

    CoreParams::Input params;

    // Create action manager
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::make_shared<OutputRegistry>();

    // Load geometry: use existing world volume or reload from geometry file
    params.geometry = build_geometry(p.model);

    if (!params.geometry->supports_safety())
    {
        CELER_LOG(warning)
            << "Geometry contains surfaces that are "
               "incompatible with the current ORANGE simple "
               "safety algorithm: multiple scattering may "
               "result in arbitrarily small steps without displacement";
    }

    // Load materials
    params.material = MaterialParams::from_import(imported);

    // Create geometry/material coupling
    params.geomaterial = GeoMaterialParams::from_import(
        imported, params.geometry, params.material);

    // Construct particle params
    params.particle = ParticleParams::from_import(imported);

    // Construct cutoffs
    params.cutoff = CutoffParams::from_import(
        imported, params.particle, params.material);

    // Construct shared data for Coulomb scattering
    params.wentzel = WentzelOKVIParams::from_import(
        imported, params.material, params.particle);

    // Load physics: create individual processes with make_shared
    params.physics = build_physics(p, params, imported);

    CELER_ASSUME(!p.field.valueless_by_exception());
    params.action_reg->insert(build_along_step(p.field, params, imported));

    // Construct RNG params
    params.rng = std::make_shared<RngParams>(p.control.seed);

    // Construct simulation params
    params.sim = std::make_shared<SimParams>([&] {
        auto input = SimParams::Input::from_import(
            imported, params.particle, p.tracking.limits.field_substeps);
        return input;
    }());

    // Number of streams
    size_type const num_streams = p.control.num_streams;
    CELER_VALIDATE(num_streams > 0,
                   << "currently p.control.num_streams must be manually set "
                      "before setup");
    params.max_streams = num_streams;

    // Construct track initialization params
    params.init = build_track_init(p.control, num_streams);

    // Set up streams
    if (auto& device = celeritas::device())
    {
        device.create_streams(num_streams);
    }

    // Number of tracks per stream
    auto tracks = p.control.capacity.tracks;
    CELER_VALIDATE(tracks > 0,
                   << "nonpositive control.capacity.tracks=" << tracks);
    params.tracks_per_stream = ceil_div(tracks, params.max_streams);

    // Construct core
    auto core_params = std::make_shared<CoreParams>(std::move(params));

    ProblemLoaded result;
    result.core_params = core_params;

    //// DIAGNOSTICS ////

    result.output_file = p.diagnostics.output_file;

    // TODO: timers, counters, perfetto_file

    if (p.diagnostics.action)
    {
        ActionDiagnostic::make_and_insert(*core_params);
    }

    if (p.diagnostics.status_checker)
    {
        // Add detailed debugging of track states
        StatusChecker::make_and_insert(*core_params);
    }

    if (p.diagnostics.step)
    {
        StepDiagnostic::make_and_insert(*core_params, p.diagnostics.step->bins);
    }

    if (p.diagnostics.slot)
    {
        SlotDiagnostic::make_and_insert(*core_params,
                                        p.diagnostics.slot->basename);
    }

    if (auto& apply = p.diagnostics.add_user_actions)
    {
        try
        {
            // Apply custom user actions
            apply(*core_params);
        }
        catch (...)
        {
            CELER_LOG(critical)
                << R"(Failed to set up user-specified diagnostics)";
            throw;
        }
    }

    //// EXPORT FILES ////

    {
        auto& ef = p.diagnostics.export_files;

        if (!ef.physics.empty())
        {
            try
            {
                RootExporter export_root(ef.physics.c_str());
                export_root(imported);
            }
            catch (RuntimeError const& e)
            {
                CELER_LOG(debug) << e.what();
                CELER_LOG(error)
                    << "Ignoring ExportFiles.physics: " << e.details().what;
            }
        }

        if (!ef.geometry.empty())
        {
            if (auto* geo
                = std::get_if<G4VPhysicalVolume const*>(&p.model.geometry))
            {
                save_gdml(*geo, ef.geometry);
            }
            else
            {
                CELER_LOG(error) << "Ignoring ExportFiles.geometry because "
                                    "the Geant4 geometry has not been loaded";
            }
        }

        // TODO: this is only implemented in accel::SharedParams, not in
        // celeritas core: should hook this into the to-be-updated
        // "primary" mechanism
        result.offload_file = ef.offload;
    }

    //// STEP COLLECTORS ////

    StepCollector::VecInterface step_interfaces;
    if (p.diagnostics.mctruth)
    {
        CELER_VALIDATE(num_streams == 1,
                       << "cannot output MC truth with multiple streams ("
                       << num_streams << " requested)");

        // Initialize ROOT file
        result.root_manager = std::make_shared<RootFileManager>(
            p.diagnostics.mctruth->output_file.c_str());

        // Create root step writer
        step_interfaces.push_back(std::make_shared<RootStepWriter>(
            result.root_manager,
            core_params->particle(),
            StepSelection::all(),
            make_write_filter(p.diagnostics.mctruth->filter)));
    }

    if (p.scoring.sd)
    {
        result.geant_sd = std::make_shared<GeantSd>(core_params->geometry(),
                                                    *core_params->particle(),
                                                    *p.scoring.sd,
                                                    core_params->max_streams());
        step_interfaces.push_back(result.geant_sd);
    }

    if (p.scoring.simple_calo)
    {
        auto simple_calo
            = std::make_shared<SimpleCalo>(p.scoring.simple_calo->volumes,
                                           *core_params->geometry(),
                                           num_streams);

        // Add to step interfaces
        step_interfaces.push_back(simple_calo);
        // Add to output interface
        core_params->output_reg()->insert(simple_calo);
    }

    if (!step_interfaces.empty())
    {
        // NOTE: step collector primarily *builds* the actions
        result.step_collector = StepCollector::make_and_insert(
            *core_params, std::move(step_interfaces));
    }

    if (p.control.optical_capacity)
    {
        result.optical_collector = build_optical_offload(
            *p.control.optical_capacity, *core_params, imported);
    }

    if (result.root_manager)
    {
        write_to_root(*core_params->action_reg(), result.root_manager.get());
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas

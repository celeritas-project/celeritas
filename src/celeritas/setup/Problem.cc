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
#include "corecel/cont/detail/VariantUtilsImpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Constant.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
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
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/optical/ScintillationParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Process.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/SimParams.hh"
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
std::shared_ptr<GeoParams> build_geometry(inp::Model const& m)
{
    auto build_from_filename
        = [](std::string const& filename) -> std::shared_ptr<GeoParams> {
        CELER_VALIDATE(!filename.empty(),
                       << "empty filename in problem.model.geometry");
        return std::make_shared<GeoParams>(filename);
    };
    auto build_from_geant
        = [&build_from_filename](G4VPhysicalVolume const* world) {
              if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
              {
                  static char const fi_hack_envname[] = "ORANGE_FORCE_INPUT";
                  auto const& filename = celeritas::getenv(fi_hack_envname);
                  if (!filename.empty())
                  {
                      CELER_LOG(warning)
                          << "Using a temporary, unsupported, and dangerous "
                             "hack to override the ORANGE geometry file: "
                          << fi_hack_envname << "='" << filename << "'";
                      return build_from_filename(filename);
                  }
              }
              CELER_VALIDATE(
                  world, << "null world pointer in problem.model.geometry");
              return std::make_shared<GeoParams>(world);
          };

    return std::visit(Overload{build_from_filename, build_from_geant},
                      m.geometry);
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
std::shared_ptr<CoreParams>
problem(inp::Problem const& p, ImportData const& imported)
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
    params.physics = [&params, p, &imported] {
        PhysicsParams::Input input;
        input.particles = params.particle;
        input.materials = params.material;
        input.action_registry = params.action_reg.get();

        // Set physics options
        input.options.fixed_step_limiter = p.tracking.force_step_limit;
        if (p.control.capacity.secondaries)
        {
            input.options.secondary_stack_factor
                = *p.control.capacity.secondaries;
        }
        else
        {
            // Default: twice the number of track slots
            input.options.secondary_stack_factor = 2.0;
        }
        input.options.spline_eloss_order = p.physics.em->eloss_spline_order;
        input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
        input.options.light.lowest_energy = ParticleOptions::Energy(
            imported.em_params.lowest_electron_energy);
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
        CELER_ASSERT(p.physics.em);
        input.processes = [&params, &em = *p.physics.em, &imported] {
            // TODO: process builder should be deleted; instead it should get
            // p.physics.em or whatever
            std::vector<std::shared_ptr<Process const>> result;
            ProcessBuilder::Options opts;
            if (em.brems)
            {
                opts.brem_combined = em.brems->combined_model;
                opts.brems_selection = [&brems = *em.brems] {
                    if (brems.rel && brems.sb)
                        return BremsModelSelection::all;
                    else if (brems.rel)
                        return BremsModelSelection::relativistic;
                    else if (brems.sb)
                        return BremsModelSelection::seltzer_berger;
                    else
                        return BremsModelSelection::none;
                }();
            }

            // TODO: add callback for user processes
            ProcessBuilder build_process(
                imported, params.particle, params.material, opts);
            for (auto pc :
                 ProcessBuilder::get_all_process_classes(imported.processes))
            {
                result.push_back(build_process(pc));
                CELER_ASSERT(result.back());
            }
            return result;
        }();

        return std::make_shared<PhysicsParams>(std::move(input));
    }();

    bool const eloss = imported.em_params.energy_loss_fluct;
    auto msc = UrbanMscParams::from_import(
        *params.particle, *params.material, imported);

    CELER_ASSUME(!p.field.valueless_by_exception());
    params.action_reg->insert(std::visit(
        return_as<std::shared_ptr<CoreStepActionInterface>>(Overload{
            [&](inp::NoField const&) {
                return AlongStepGeneralLinearAction::from_params(
                    params.action_reg->next_id(),
                    *params.material,
                    *params.particle,
                    msc,
                    eloss);
            },
            [&](inp::UniformField const& field) {
                UniformFieldParams field_params;

                if (field.units != UnitSystem::si)
                {
                    CELER_NOT_IMPLEMENTED("field units in other unit systems");
                }
                field_params.field = field.strength;
                field_params.options = field.driver_options;

                // Interpret input in units of Tesla
                for (real_type& v : field_params.field)
                {
                    v = native_value_from(units::FieldTesla{v});
                }

                return AlongStepUniformMscAction::from_params(
                    params.action_reg->next_id(),
                    *params.material,
                    *params.particle,
                    field_params,
                    msc,
                    eloss);
            },
            [](inp::RZMapField const&)
                -> std::shared_ptr<CoreStepActionInterface> {
                CELER_NOT_IMPLEMENTED("building RZ map field through input");
            },
        }),
        p.field));

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
    params.init = [&] {
        CELER_VALIDATE(p.control.capacity.initializers > 0,
                       << "nonpositive capacity.initializers="
                       << p.control.capacity.initializers);
        CELER_VALIDATE(p.control.capacity.events > 0,
                       << "nonpositive capacity.events="
                       << p.control.capacity.events);
        TrackInitParams::Input input;
        input.capacity = ceil_div(p.control.capacity.initializers, num_streams);
        input.max_events = p.control.capacity.events;
        if (p.control.track_order)
        {
            input.track_order = *p.control.track_order;
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
            CELER_LOG(debug)
                << "Set default track order " << to_cstring(input.track_order);
        }

        return std::make_shared<TrackInitParams>(std::move(input));
    }();

    // Number of tracks per stream
    auto tracks = p.control.capacity.tracks;
    CELER_VALIDATE(tracks > 0,
                   << "nonpositive control.capacity.tracks=" << tracks);
    params.tracks_per_stream = ceil_div(tracks, params.max_streams);

    // Construct core
    auto core_params = std::make_shared<CoreParams>(std::move(params));

    //// DIAGNOSTICS ////

    if (p.diagnostics.action)
    {
        ActionDiagnostic::make_and_insert(*core_params);
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

    //// STEP COLLECTORS ////

    StepCollector::VecInterface step_interfaces;
    std::shared_ptr<RootFileManager> root_manager;
    if (p.diagnostics.mctruth)
    {
        CELER_VALIDATE(num_streams == 1,
                       << "cannot output MC truth with multiple streams ("
                       << num_streams << " requested)");

        // Initialize ROOT file
        root_manager = std::make_shared<RootFileManager>(
            p.diagnostics.mctruth->output_file.c_str());

        // Create root step writer
        step_interfaces.push_back(std::make_shared<RootStepWriter>(
            root_manager,
            core_params->particle(),
            StepSelection::all(),
            make_write_filter(p.diagnostics.mctruth->filter)));
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
        // TODO: step collector really just *builds* the actions: it's ok that
        // it immediately goes out of scope
        StepCollector(core_params->geometry(),
                      std::move(step_interfaces),
                      core_params->aux_reg().get(),
                      core_params->action_reg().get());
    }

    if (p.control.optical_capacity)
    {
        CELER_EXPECT(core_params);

        using optical::CherenkovParams;
        using optical::MaterialParams;
        using optical::ScintillationParams;

        CELER_VALIDATE(
            !imported.optical_materials.empty(),
            << R"(an optical tracking loop was requested but no optical materials are present)");

        OpticalCollector::Input oc_inp;
        oc_inp.material = MaterialParams::from_import(
            imported, *core_params->geomaterial(), *core_params->material());
        oc_inp.cherenkov = std::make_shared<CherenkovParams>(*oc_inp.material);
        oc_inp.scintillation = ScintillationParams::from_import(
            imported, core_params->particle());

        // Map from optical capacity
        auto const& optical_capacity = *p.control.optical_capacity;
        oc_inp.num_track_slots = ceil_div(optical_capacity.tracks, num_streams);
        oc_inp.buffer_capacity
            = ceil_div(optical_capacity.generators, num_streams);
        oc_inp.initializer_capacity
            = ceil_div(optical_capacity.initializers, num_streams);
        oc_inp.auto_flush = ceil_div(optical_capacity.primaries, num_streams);

        CELER_ASSERT(oc_inp);

        // TODO: optical collector really just *builds* the optical setup: it's
        // ok that it immediately goes out of scope
        OpticalCollector(*core_params, std::move(oc_inp));
    }

    if (root_manager)
    {
        write_to_root(*core_params->action_reg(), root_manager.get());
    }
    return core_params;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas

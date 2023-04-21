//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PrimaryGeneratorOptionsIO.json.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/RootStepWriterIO.json.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, EnergyDiagInput const& v)
{
    j = nlohmann::json{{"axis", std::string(1, v.axis)},
                       {"min", v.min},
                       {"max", v.max},
                       {"num_bins", v.num_bins}};
}

void from_json(nlohmann::json const& j, EnergyDiagInput& v)
{
    std::string temp_axis;
    j.at("axis").get_to(temp_axis);
    CELER_VALIDATE(temp_axis.size() == 1,
                   << "axis spec has length " << temp_axis.size()
                   << " (must be a single character)");
    v.axis = temp_axis.front();
    j.at("min").get_to(v.min);
    j.at("max").get_to(v.max);
    j.at("num_bins").get_to(v.num_bins);
}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, LDemoArgs& v)
{
#define LDIO_LOAD_OPTION(NAME)          \
    do                                  \
    {                                   \
        if (j.contains(#NAME))          \
            j.at(#NAME).get_to(v.NAME); \
    } while (0)
#define LDIO_LOAD_REQUIRED(NAME) j.at(#NAME).get_to(v.NAME)

    LDIO_LOAD_OPTION(cuda_heap_size);
    LDIO_LOAD_OPTION(cuda_stack_size);
    LDIO_LOAD_OPTION(environ);

    LDIO_LOAD_REQUIRED(geometry_filename);
    LDIO_LOAD_REQUIRED(physics_filename);
    LDIO_LOAD_OPTION(hepmc3_filename);
    LDIO_LOAD_OPTION(mctruth_filename);

    LDIO_LOAD_OPTION(mctruth_filter);
    LDIO_LOAD_OPTION(primary_gen_options);

    LDIO_LOAD_OPTION(seed);
    LDIO_LOAD_OPTION(max_num_tracks);
    LDIO_LOAD_OPTION(max_steps);
    LDIO_LOAD_REQUIRED(initializer_capacity);
    LDIO_LOAD_REQUIRED(max_events);
    LDIO_LOAD_REQUIRED(secondary_stack_factor);
    LDIO_LOAD_REQUIRED(enable_diagnostics);
    LDIO_LOAD_REQUIRED(use_device);
    LDIO_LOAD_OPTION(sync);

    LDIO_LOAD_OPTION(mag_field);
    LDIO_LOAD_OPTION(field_options);

    LDIO_LOAD_OPTION(step_limiter);
    LDIO_LOAD_OPTION(brem_combined);
    LDIO_LOAD_OPTION(energy_diag);
    LDIO_LOAD_OPTION(track_order);
    LDIO_LOAD_OPTION(geant_options);

#undef LDIO_LOAD_OPTION
#undef LDIO_LOAD_REQUIRED

    CELER_VALIDATE(v.hepmc3_filename.empty() != !v.primary_gen_options,
                   << "either a HepMC3 filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(!v.mctruth_filter || !v.mctruth_filename.empty(),
                   << "'mctruth_filter' cannot be specified without providing "
                      "'mctruth_filename'");
    CELER_VALIDATE(v.mag_field == LDemoArgs::no_field()
                       || !j.contains("field_options"),
                   << "'field_options' cannot be specified without providing "
                      "'mag_field'");
}

//---------------------------------------------------------------------------//
/*!
 * Save options to JSON.
 */
void to_json(nlohmann::json& j, LDemoArgs const& v)
{
    j = nlohmann::json::object();
    LDemoArgs const default_args;
#define LDIO_SAVE_OPTION(NAME)           \
    do                                   \
    {                                    \
        if (v.NAME != default_args.NAME) \
            j[#NAME] = v.NAME;           \
    } while (0)
#define LDIO_SAVE_REQUIRED(NAME) j[#NAME] = v.NAME

    LDIO_SAVE_OPTION(cuda_heap_size);
    LDIO_SAVE_OPTION(cuda_stack_size);
    LDIO_SAVE_REQUIRED(environ);

    LDIO_SAVE_REQUIRED(geometry_filename);
    LDIO_SAVE_REQUIRED(physics_filename);
    LDIO_SAVE_OPTION(hepmc3_filename);
    LDIO_SAVE_OPTION(mctruth_filename);

    if (!v.mctruth_filename.empty())
    {
        LDIO_SAVE_REQUIRED(mctruth_filter);
    }
    if (v.hepmc3_filename.empty())
    {
        LDIO_SAVE_REQUIRED(primary_gen_options);
    }

    LDIO_SAVE_OPTION(seed);
    LDIO_SAVE_OPTION(max_num_tracks);
    LDIO_SAVE_OPTION(max_steps);
    LDIO_SAVE_REQUIRED(initializer_capacity);
    LDIO_SAVE_REQUIRED(max_events);
    LDIO_SAVE_REQUIRED(secondary_stack_factor);
    LDIO_SAVE_REQUIRED(enable_diagnostics);
    LDIO_SAVE_REQUIRED(use_device);
    LDIO_SAVE_OPTION(sync);

    LDIO_SAVE_OPTION(mag_field);
    if (v.mag_field != LDemoArgs::no_field())
    {
        LDIO_SAVE_REQUIRED(field_options);
    }

    LDIO_SAVE_OPTION(step_limiter);
    LDIO_SAVE_OPTION(brem_combined);

    if (v.enable_diagnostics)
    {
        LDIO_SAVE_REQUIRED(energy_diag);
    }
    LDIO_SAVE_OPTION(track_order);
    if (ends_with(v.physics_filename, ".gdml"))
    {
        LDIO_SAVE_REQUIRED(geant_options);
    }

#undef LDIO_SAVE_OPTION
#undef LDIO_SAVE_REQUIRED
}

//---------------------------------------------------------------------------//
std::shared_ptr<CoreParams>
build_core_params(LDemoArgs const& args, std::shared_ptr<OutputRegistry> outreg)
{
    CELER_LOG(status) << "Loading input and initializing problem data";
    ScopedMem record_mem("demo_loop.load_core_params");
    CoreParams::Input params;
    ImportData const imported = [&args] {
        if (ends_with(args.physics_filename, ".root"))
        {
            // Load imported from ROOT file
            return RootImporter(args.physics_filename.c_str())();
        }
        else if (ends_with(args.physics_filename, ".gdml"))
        {
            // Load imported directly from Geant4
            return GeantImporter(
                GeantSetup(args.physics_filename, args.geant_options))();
        }
        CELER_VALIDATE(false,
                       << "invalid physics filename '" << args.physics_filename
                       << "' (expected gdml or root)");
    }();

    // Create action manager
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::move(outreg);

    // Load geometry
    params.geometry
        = std::make_shared<GeoParams>(args.geometry_filename.c_str());
    if (!params.geometry->supports_safety())
    {
        CELER_LOG(warning) << "Geometry contains surfaces that are "
                              "incompatible with the current ORANGE simple "
                              "safety algorithm: multiple scattering may "
                              "result in arbitrarily small steps";
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

    // Load physics: create individual processes with make_shared
    params.physics = [&params, &args, &imported] {
        PhysicsParams::Input input;
        input.particles = params.particle;
        input.materials = params.material;
        input.action_registry = params.action_reg.get();

        input.options.fixed_step_limiter = args.step_limiter;
        input.options.secondary_stack_factor = args.secondary_stack_factor;
        input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
        input.options.lowest_electron_energy = PhysicsParamsOptions::Energy{
            imported.em_params.lowest_electron_energy};

        input.processes = [&params, &args, &imported] {
            std::vector<std::shared_ptr<Process const>> result;
            ProcessBuilder::Options opts;
            opts.brem_combined = args.brem_combined;

            ProcessBuilder build_process(
                imported, params.particle, params.material, opts);
            for (auto p :
                 ProcessBuilder::get_all_process_classes(imported.processes))
            {
                result.push_back(build_process(p));
                CELER_ASSERT(result.back());
            }
            return result;
        }();

        return std::make_shared<PhysicsParams>(std::move(input));
    }();

    bool eloss = imported.em_params.energy_loss_fluct;
    auto msc = UrbanMscParams::from_import(
        *params.particle, *params.material, imported);
    if (args.mag_field == LDemoArgs::no_field())
    {
        // Create along-step action
        auto along_step = AlongStepGeneralLinearAction::from_params(
            params.action_reg->next_id(),
            *params.material,
            *params.particle,
            msc,
            eloss);
        params.action_reg->insert(along_step);
    }
    else
    {
        CELER_VALIDATE(!eloss,
                       << "energy loss fluctuations are not supported "
                          "simultaneoulsy with magnetic field");
        UniformFieldParams field_params;
        field_params.field = args.mag_field;
        field_params.options = args.field_options;

        // Interpret input in units of Tesla
        for (real_type& f : field_params.field)
        {
            f *= units::tesla;
        }

        auto along_step = std::make_shared<AlongStepUniformMscAction>(
            params.action_reg->next_id(), field_params, msc);
        CELER_ASSERT(along_step->field() != LDemoArgs::no_field());
        params.action_reg->insert(along_step);
    }

    // Construct RNG params
    params.rng = std::make_shared<RngParams>(args.seed);

    // Construct simulation params
    params.sim = SimParams::from_import(imported, params.particle);

    // Construct track initialization params
    params.init = [&args] {
        CELER_VALIDATE(args.initializer_capacity > 0,
                       << "nonpositive initializer_capacity="
                       << args.initializer_capacity);
        CELER_VALIDATE(args.max_events > 0,
                       << "nonpositive max_events=" << args.max_events);
        CELER_VALIDATE(
            !args.primary_gen_options
                || args.max_events >= args.primary_gen_options.num_events,
            << "max_events=" << args.max_events
            << " cannot be less than num_events="
            << args.primary_gen_options.num_events);
        TrackInitParams::Input input;
        input.capacity = args.initializer_capacity;
        input.max_events = args.max_events;
        input.track_order = args.track_order;
        return std::make_shared<TrackInitParams>(std::move(input));
    }();

    return std::make_shared<CoreParams>(std::move(params));
}

//---------------------------------------------------------------------------//
/*!
 * Construct parameters, input, and transporter from the given run arguments.
 */
std::unique_ptr<TransporterBase>
build_transporter(LDemoArgs const& args,
                  std::shared_ptr<CoreParams const> params)
{
    using celeritas::MemSpace;

    // Save constants from args
    TransporterInput input;
    CELER_VALIDATE(args.max_num_tracks > 0,
                   << "nonpositive max_num_tracks=" << args.max_num_tracks);
    CELER_VALIDATE(args.max_steps > 0,
                   << "nonpositive max_steps=" << args.max_steps);
    input.num_track_slots = args.max_num_tracks;
    input.max_steps = args.max_steps;
    input.enable_diagnostics = args.enable_diagnostics;
    input.sync = args.sync;
    input.energy_diag = args.energy_diag;

    // Create core params
    input.params = std::move(params);

    std::unique_ptr<TransporterBase> result;

    if (args.use_device)
    {
        CELER_VALIDATE(celeritas::device(),
                       << "CUDA device is unavailable but GPU run was "
                          "requested");
        result = std::make_unique<Transporter<MemSpace::device>>(
            std::move(input));
    }
    else
    {
        result
            = std::make_unique<Transporter<MemSpace::host>>(std::move(input));
    }
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace demo_loop

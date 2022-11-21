//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <algorithm>
#include <set>
#include <string>

#include "corecel/cont/Array.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh" // IWYU pragma: keep
#include "celeritas/global/ActionRegistry.hh"
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
#include "celeritas/track/TrackInitParams.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, const EnergyDiagInput& v)
{
    j = nlohmann::json{{"axis", std::string(1, v.axis)},
                       {"min", v.min},
                       {"max", v.max},
                       {"num_bins", v.num_bins}};
}

void from_json(const nlohmann::json& j, EnergyDiagInput& v)
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

namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Get the set of all process classes in the input
auto get_all_process_classes(const std::vector<ImportProcess>& processes)
    -> decltype(auto)
{
    std::set<ImportProcessClass> result;
    for (const auto& p : processes)
    {
        result.insert(p.process_class);
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON and ROOT
void to_json(nlohmann::json& j, const LDemoArgs& v)
{
    j = nlohmann::json{{"geometry_filename", v.geometry_filename},
                       {"physics_filename", v.physics_filename},
                       {"seed", v.seed},
                       {"max_num_tracks", v.max_num_tracks},
                       {"max_steps", v.max_steps},
                       {"initializer_capacity", v.initializer_capacity},
                       {"max_events", v.max_events},
                       {"secondary_stack_factor", v.secondary_stack_factor},
                       {"enable_diagnostics", v.enable_diagnostics},
                       {"use_device", v.use_device},
                       {"sync", v.sync},
                       {"mag_field", v.mag_field},
                       {"brem_combined", v.brem_combined}};
    if (v.mag_field != LDemoArgs::no_field())
    {
        j["field_options"] = v.field_options;
    }
    if (v.enable_diagnostics)
    {
        j["energy_diag"] = v.energy_diag;
    }
    if (v.step_limiter > 0)
    {
        j["step_limiter"] = v.step_limiter;
    }
    if (ends_with(v.physics_filename, ".gdml"))
    {
        j["geant_options"] = v.geant_options;
    }
    if (v.primary_gen_options)
    {
        j["primary_gen_options"] = v.primary_gen_options;
    }
    if (!v.hepmc3_filename.empty())
    {
        j["hepmc3_filename"] = v.hepmc3_filename;
    }
    if (!v.mctruth_filename.empty())
    {
        j["mctruth_filename"] = v.mctruth_filename;
    }
}

void from_json(const nlohmann::json& j, LDemoArgs& v)
{
    j.at("geometry_filename").get_to(v.geometry_filename);
    j.at("physics_filename").get_to(v.physics_filename);
    if (j.contains("hepmc3_filename"))
    {
        j.at("hepmc3_filename").get_to(v.hepmc3_filename);
    }
    if (j.contains("mctruth_filename"))
    {
        j.at("mctruth_filename").get_to(v.mctruth_filename);
    }
    if (j.contains("primary_gen_options"))
    {
        j.at("primary_gen_options").get_to(v.primary_gen_options);
    }
    CELER_VALIDATE(v.hepmc3_filename.empty() != !v.primary_gen_options,
                   << "either a HepMC3 filename or options to generate "
                      "primaries must be provided (but not both)");

    j.at("seed").get_to(v.seed);
    j.at("max_num_tracks").get_to(v.max_num_tracks);
    if (j.contains("max_steps"))
    {
        j.at("max_steps").get_to(v.max_steps);
    }
    j.at("initializer_capacity").get_to(v.initializer_capacity);
    j.at("max_events").get_to(v.max_events);
    j.at("secondary_stack_factor").get_to(v.secondary_stack_factor);
    j.at("enable_diagnostics").get_to(v.enable_diagnostics);
    j.at("use_device").get_to(v.use_device);
    j.at("sync").get_to(v.sync);
    if (j.contains("mag_field"))
    {
        j.at("mag_field").get_to(v.mag_field);
    }
    if (v.mag_field != LDemoArgs::no_field() && j.contains("field_options"))
    {
        j.at("field_options").get_to(v.field_options);
    }
    if (j.contains("step_limiter"))
    {
        j.at("step_limiter").get_to(v.step_limiter);
    }

    j.at("brem_combined").get_to(v.brem_combined);

    if (j.contains("energy_diag"))
    {
        j.at("energy_diag").get_to(v.energy_diag);
    }

    if (j.contains("geant_options"))
    {
        j.at("geant_options").get_to(v.geant_options);
    }
}
//!@}

//---------------------------------------------------------------------------//
TransporterInput load_input(const LDemoArgs& args)
{
    CELER_LOG(status) << "Loading input and initializing problem data";
    TransporterInput  result;
    CoreParams::Input params;

    ImportData imported_data;
    if (ends_with(args.physics_filename, ".root"))
    {
        // Load imported_data from ROOT file
        imported_data = RootImporter(args.physics_filename.c_str())();
    }
    else if (ends_with(args.physics_filename, ".gdml"))
    {
        // Load imported_data directly from Geant4
        imported_data = GeantImporter(
            GeantSetup(args.physics_filename, args.geant_options))();
    }
    else
    {
        CELER_VALIDATE(false,
                       << "invalid physics filename '" << args.physics_filename
                       << "' (expected gdml or root)");
    }

    // Create action manager
    {
        params.action_reg = std::make_shared<ActionRegistry>();
    }

    // Load geometry
    {
        params.geometry
            = std::make_shared<GeoParams>(args.geometry_filename.c_str());
        if (!params.geometry->supports_safety())
        {
            CELER_LOG(warning)
                << "Geometry contains surfaces that are "
                   "incompatible with the current ORANGE simple "
                   "safety algorithm: multiple scattering may "
                   "result in arbitrarily small steps";
        }
    }

    // Load materials
    {
        params.material = MaterialParams::from_import(imported_data);
    }

    // Create geometry/material coupling
    {
        params.geomaterial = GeoMaterialParams::from_import(
            imported_data, params.geometry, params.material);
    }

    // Construct particle params
    {
        params.particle = ParticleParams::from_import(imported_data);
    }

    // Construct cutoffs
    {
        params.cutoff = CutoffParams::from_import(
            imported_data, params.particle, params.material);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles       = params.particle;
        input.materials       = params.material;
        input.action_registry = params.action_reg.get();

        input.options.fixed_step_limiter     = args.step_limiter;
        input.options.secondary_stack_factor = args.secondary_stack_factor;
        input.options.linear_loss_limit
            = imported_data.em_params.linear_loss_limit;

        {
            ProcessBuilder::Options opts;
            opts.brem_combined = args.brem_combined;

            ProcessBuilder build_process(
                imported_data, opts, params.particle, params.material);
            // TODO: there's got to be a cleaner way to get the set of all
            // processes: maybe better to change how the ImportData is
            // structured
            for (auto p : get_all_process_classes(imported_data.processes))
            {
                input.processes.push_back(build_process(p));
            }
        }

        params.physics = std::make_shared<PhysicsParams>(std::move(input));
    }

    bool eloss = imported_data.em_params.energy_loss_fluct;
    if (args.mag_field == LDemoArgs::no_field())
    {
        // Create along-step action
        auto along_step = AlongStepGeneralLinearAction::from_params(
            params.action_reg->next_id(),
            *params.material,
            *params.particle,
            *params.physics,
            eloss);
        params.action_reg->insert(along_step);
    }
    else
    {
        CELER_VALIDATE(!eloss,
                       << "energy loss fluctuations are not supported "
                          "simultaneoulsy with magnetic field");
        UniformFieldParams field_params;
        field_params.field   = args.mag_field;
        field_params.options = args.field_options;

        // Interpret input in units of Tesla
        for (real_type& f : field_params.field)
        {
            f *= units::tesla;
        }

        auto along_step = AlongStepUniformMscAction::from_params(
            params.action_reg->next_id(), *params.physics, field_params);
        CELER_ASSERT(along_step->field() != LDemoArgs::no_field());
        params.action_reg->insert(along_step);
    }

    // Construct RNG params
    {
        params.rng = std::make_shared<RngParams>(args.seed);
    }

    // Construct track initialization params
    {
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
        input.capacity   = args.initializer_capacity;
        input.max_events = args.max_events;
        params.init      = std::make_shared<TrackInitParams>(input);
    }

    // Create params
    CELER_ASSERT(params);
    result.params = std::make_shared<CoreParams>(std::move(params));

    // Save constants
    CELER_VALIDATE(args.max_num_tracks > 0,
                   << "nonpositive max_num_tracks=" << args.max_num_tracks);
    CELER_VALIDATE(args.max_steps > 0,
                   << "nonpositive max_steps=" << args.max_steps);
    result.num_track_slots    = args.max_num_tracks;
    result.max_steps          = args.max_steps;
    result.enable_diagnostics = args.enable_diagnostics;
    result.sync               = args.sync;

    // Save diagnosics
    result.energy_diag = args.energy_diag;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct parameters, input, and transporter from the given run arguments.
 */
std::unique_ptr<TransporterBase> build_transporter(const LDemoArgs& run_args)
{
    using celeritas::MemSpace;

    TransporterInput                 input = load_input(run_args);
    std::unique_ptr<TransporterBase> result;

    if (run_args.use_device)
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
} // namespace demo_loop

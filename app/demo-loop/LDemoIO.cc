//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <algorithm>
#include <string>

#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/process/BremsstrahlungProcess.hh"
#include "celeritas/em/process/ComptonProcess.hh"
#include "celeritas/em/process/EIonizationProcess.hh"
#include "celeritas/em/process/EPlusAnnihilationProcess.hh"
#include "celeritas/em/process/GammaConversionProcess.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/em/process/PhotoelectricProcess.hh"
#include "celeritas/em/process/RayleighProcess.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetupOptionsIO.json.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ImportedProcessAdapter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/random/RngParams.hh"

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
//! Check that volume names are consistent between the ROOT file and geometry
bool volumes_are_consistent(const GeoParams&                 geo,
                            const std::vector<ImportVolume>& imported)
{
    return geo.num_volumes() == imported.size()
           && std::all_of(RangeIter<VolumeId>(VolumeId{0}),
                          RangeIter<VolumeId>(VolumeId{geo.num_volumes()}),
                          [&](VolumeId vol) {
                              return geo.id_to_label(vol)
                                     == Label::from_geant(
                                         imported[vol.unchecked_get()].name);
                          });
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, const LDemoArgs& v)
{
    j = nlohmann::json{{"geometry_filename", v.geometry_filename},
                       {"physics_filename", v.physics_filename},
                       {"hepmc3_filename", v.hepmc3_filename},
                       {"seed", v.seed},
                       {"max_num_tracks", v.max_num_tracks},
                       {"max_steps", v.max_steps},
                       {"initializer_capacity", v.initializer_capacity},
                       {"secondary_stack_factor", v.secondary_stack_factor},
                       {"enable_diagnostics", v.enable_diagnostics},
                       {"use_device", v.use_device},
                       {"sync", v.sync},
                       {"rayleigh", v.rayleigh},
                       {"eloss_fluctuation", v.eloss_fluctuation},
                       {"brem_combined", v.brem_combined},
                       {"brem_lpm", v.brem_lpm},
                       {"conv_lpm", v.conv_lpm},
                       {"enable_msc", v.enable_msc}};
    if (v.enable_diagnostics)
    {
        j["energy_diag"] = v.energy_diag;
    }
    if (v.step_limiter > 0)
    {
        j["step_limiter"] = v.step_limiter;
    }
    if (ends_with(v.geometry_filename, ".gdml"))
    {
        j["geant_options"] = v.geant_options;
    }
}

void from_json(const nlohmann::json& j, LDemoArgs& v)
{
    j.at("geometry_filename").get_to(v.geometry_filename);
    j.at("physics_filename").get_to(v.physics_filename);
    j.at("hepmc3_filename").get_to(v.hepmc3_filename);

    j.at("seed").get_to(v.seed);
    j.at("max_num_tracks").get_to(v.max_num_tracks);
    if (j.contains("max_steps"))
    {
        j.at("max_steps").get_to(v.max_steps);
    }
    j.at("initializer_capacity").get_to(v.initializer_capacity);
    j.at("secondary_stack_factor").get_to(v.secondary_stack_factor);
    j.at("enable_diagnostics").get_to(v.enable_diagnostics);
    j.at("use_device").get_to(v.use_device);
    j.at("sync").get_to(v.sync);
    if (j.contains("step_limiter"))
    {
        j.at("step_limiter").get_to(v.step_limiter);
    }

    j.at("rayleigh").get_to(v.rayleigh);
    j.at("eloss_fluctuation").get_to(v.eloss_fluctuation);
    j.at("brem_combined").get_to(v.brem_combined);
    j.at("brem_lpm").get_to(v.brem_lpm);
    j.at("conv_lpm").get_to(v.conv_lpm);
    j.at("enable_msc").get_to(v.enable_msc);

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
        ActionManager::Options opts;
        opts.sync         = args.sync;
        params.action_mgr = std::make_shared<ActionManager>(opts);
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
        GeoMaterialParams::Input input;
        input.geometry  = params.geometry;
        input.materials = params.material;

        input.volume_to_mat.resize(imported_data.volumes.size());
        for (auto volume_idx :
             range<VolumeId::size_type>(input.volume_to_mat.size()))
        {
            input.volume_to_mat[volume_idx]
                = MaterialId{imported_data.volumes[volume_idx].material_id};
        }
        if (!volumes_are_consistent(*input.geometry, imported_data.volumes))
        {
            // Volume names do not match exactly between exported ROOT file and
            // the geometry (possibly using a ROOT/GDML input with an ORANGE
            // geometry): try to let the GeoMaterialParams remap them
            CELER_LOG(warning) << "Volume/material mapping is inconsistent "
                                  "between Geant4 data and geometry file: "
                                  "attempting to remap";
            input.volume_labels.resize(imported_data.volumes.size());
            for (auto volume_idx : range(imported_data.volumes.size()))
            {
                input.volume_labels[volume_idx] = Label::from_geant(
                    imported_data.volumes[volume_idx].name);
            }
        }
        params.geomaterial
            = std::make_shared<GeoMaterialParams>(std::move(input));
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
        input.particles = params.particle;
        input.materials = params.material;
        input.options.fixed_step_limiter     = args.step_limiter;
        input.options.secondary_stack_factor = args.secondary_stack_factor;
        input.action_manager                 = params.action_mgr.get();

        BremsstrahlungProcess::Options brem_options;
        brem_options.combined_model  = args.brem_combined;
        brem_options.enable_lpm      = args.brem_lpm;
        brem_options.use_integral_xs = true;

        GammaConversionProcess::Options conv_options;
        conv_options.enable_lpm = args.conv_lpm;

        EPlusAnnihilationProcess::Options epgg_options;
        epgg_options.use_integral_xs = true;

        EIonizationProcess::Options ioni_options;
        ioni_options.use_integral_xs = true;

        auto process_data = std::make_shared<ImportedProcesses>(
            std::move(imported_data.processes));
        input.processes.push_back(
            std::make_shared<ComptonProcess>(params.particle, process_data));
        input.processes.push_back(std::make_shared<PhotoelectricProcess>(
            params.particle, params.material, process_data));
        if (args.rayleigh)
        {
            input.processes.push_back(std::make_shared<RayleighProcess>(
                params.particle, params.material, process_data));
        }
        input.processes.push_back(std::make_shared<GammaConversionProcess>(
            params.particle, process_data, conv_options));
        input.processes.push_back(std::make_shared<EPlusAnnihilationProcess>(
            params.particle, epgg_options));
        input.processes.push_back(std::make_shared<EIonizationProcess>(
            params.particle, process_data, ioni_options));
        input.processes.push_back(std::make_shared<BremsstrahlungProcess>(
            params.particle, params.material, process_data, brem_options));
        if (args.enable_msc)
        {
            input.processes.push_back(
                std::make_shared<MultipleScatteringProcess>(
                    params.particle, params.material, process_data));
        }
        params.physics = std::make_shared<PhysicsParams>(std::move(input));
    }
    {
        // Create along-step action
        params.along_step = AlongStepGeneralLinearAction::from_params(
            *params.material,
            *params.particle,
            *params.physics,
            args.eloss_fluctuation,
            params.action_mgr.get());
    }

    // Construct RNG params
    {
        params.rng = std::make_shared<RngParams>(args.seed);
    }

    // Create params
    CELER_ASSERT(params);
    result.params = std::make_shared<CoreParams>(std::move(params));

    // Save constants
    CELER_VALIDATE(args.max_num_tracks > 0,
                   << "nonpositive max_num_tracks=" << args.max_num_tracks);
    CELER_VALIDATE(args.initializer_capacity > 0,
                   << "nonpositive initializer_capacity="
                   << args.initializer_capacity);
    CELER_VALIDATE(args.max_steps > 0,
                   << "nonpositive max_steps=" << args.max_steps);
    result.num_track_slots    = args.max_num_tracks;
    result.num_initializers   = args.initializer_capacity;
    result.max_steps          = args.max_steps;
    result.enable_diagnostics = args.enable_diagnostics;

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

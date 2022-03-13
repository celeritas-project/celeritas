//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoIO.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <algorithm>
#include <string>

#include "comm/Logger.hh"
#include "geometry/GeoMaterialParams.hh"
#include "geometry/GeoParams.hh"
#include "io/EventReader.hh"
#include "io/ImportData.hh"
#include "io/RootImporter.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/em/BremsstrahlungProcess.hh"
#include "physics/em/ComptonProcess.hh"
#include "physics/em/EIonizationProcess.hh"
#include "physics/em/EPlusAnnihilationProcess.hh"
#include "physics/em/GammaConversionProcess.hh"
#include "physics/em/PhotoelectricProcess.hh"
#include "physics/em/RayleighProcess.hh"
#include "physics/material/MaterialParams.hh"
#include "random/RngParams.hh"

using namespace celeritas;

namespace celeritas
{
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
} // namespace celeritas

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Check that volume names are consistent between the ROOT file and geometry
bool volumes_are_consistent(const GeoParams&                 geo,
                            const std::vector<ImportVolume>& imported_data)
{
    return geo.num_volumes() == imported_data.size()
           && std::all_of(RangeIter<VolumeId>(VolumeId{0}),
                          RangeIter<VolumeId>(VolumeId{geo.num_volumes()}),
                          [&](VolumeId vol) {
                              return geo.id_to_label(vol)
                                     == imported_data[vol.unchecked_get()].name;
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
                       {"brem_lpm", v.brem_lpm}};
    if (v.enable_diagnostics)
    {
        j["energy_diag"] = v.energy_diag;
    }
}

void from_json(const nlohmann::json& j, LDemoArgs& v)
{
    j.at("geometry_filename").get_to(v.geometry_filename);
    j.at("physics_filename").get_to(v.physics_filename);
    j.at("hepmc3_filename").get_to(v.hepmc3_filename);
    j.at("rayleigh").get_to(v.rayleigh);
    j.at("eloss_fluctuation").get_to(v.eloss_fluctuation);
    j.at("brem_combined").get_to(v.brem_combined);
    j.at("brem_lpm").get_to(v.brem_lpm);
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

    if (j.contains("energy_diag"))
    {
        j.at("energy_diag").get_to(v.energy_diag);
    }
}
//!@}

//---------------------------------------------------------------------------//
TransporterInput load_input(const LDemoArgs& args)
{
    CELER_LOG(status) << "Loading input and initializing problem data";
    TransporterInput result;

    // Load imported_data from ROOT file
    auto imported_data = RootImporter(args.physics_filename.c_str())();

    // Load geometry
    {
        result.geometry
            = std::make_shared<GeoParams>(args.geometry_filename.c_str());
    }

    // Load materials
    {
        result.materials = MaterialParams::from_import(imported_data);
    }

    // Create geometry/material coupling
    {
        GeoMaterialParams::Input input;
        input.geometry  = result.geometry;
        input.materials = result.materials;

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
                                  "between ROOT file and geometry file: "
                                  "attempting to remap";
            input.volume_names.resize(imported_data.volumes.size());
            for (auto volume_idx : range(imported_data.volumes.size()))
            {
                input.volume_names[volume_idx]
                    = std::move(imported_data.volumes[volume_idx].name);
            }
        }
        result.geo_mats = std::make_shared<GeoMaterialParams>(std::move(input));
    }

    // Construct particle params
    {
        result.particles = ParticleParams::from_import(imported_data);
    }

    // Construct cutoffs
    {
        result.cutoffs = CutoffParams::from_import(
            imported_data, result.particles, result.materials);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles                  = result.particles;
        input.materials                  = result.materials;
        input.options.enable_fluctuation = args.eloss_fluctuation;

        BremsstrahlungProcess::Options brem_options;
        brem_options.combined_model = args.brem_combined;
        brem_options.enable_lpm     = args.brem_lpm;

        auto process_data = std::make_shared<ImportedProcesses>(
            std::move(imported_data.processes));
        input.processes.push_back(
            std::make_shared<ComptonProcess>(result.particles, process_data));
        input.processes.push_back(std::make_shared<PhotoelectricProcess>(
            result.particles, result.materials, process_data));
        if (args.rayleigh)
        {
            input.processes.push_back(std::make_shared<RayleighProcess>(
                result.particles, result.materials, process_data));
        }
        input.processes.push_back(std::make_shared<GammaConversionProcess>(
            result.particles, process_data));
        input.processes.push_back(
            std::make_shared<EPlusAnnihilationProcess>(result.particles));
        input.processes.push_back(std::make_shared<EIonizationProcess>(
            result.particles, process_data));
        input.processes.push_back(std::make_shared<BremsstrahlungProcess>(
            result.particles, result.materials, process_data, brem_options));

        result.physics = std::make_shared<PhysicsParams>(std::move(input));
    }

    // Construct RNG params
    {
        result.rng = std::make_shared<RngParams>(args.seed);
    }

    // Save constants
    result.max_num_tracks         = args.max_num_tracks;
    result.max_steps              = args.max_steps;
    result.secondary_stack_factor = args.secondary_stack_factor;
    result.enable_diagnostics     = args.enable_diagnostics;
    result.sync                   = args.sync;

    // Propagate diagnosics
    result.energy_diag = args.energy_diag;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Load primary particles from the demo input arguments.
 */
std::shared_ptr<celeritas::TrackInitParams>
load_primaries(const std::shared_ptr<const celeritas::ParticleParams>& particles,
               const LDemoArgs&                                        args)
{
    CELER_EXPECT(particles);
    EventReader read_all_events(args.hepmc3_filename.c_str(), particles);
    TrackInitParams::Input input;
    input.primaries = read_all_events();
    input.capacity  = args.initializer_capacity;
    return std::make_shared<TrackInitParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct parameters, input, and transporter from the given run arguments.
 */
std::unique_ptr<TransporterBase> build_transporter(const LDemoArgs& run_args)
{
    using celeritas::MemSpace;
    using celeritas::Transporter;
    using celeritas::TransporterInput;

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

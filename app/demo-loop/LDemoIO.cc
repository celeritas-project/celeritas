//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoIO.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

#include <algorithm>
#include "comm/Logger.hh"
#include "geometry/GeoMaterialParams.hh"
#include "geometry/GeoParams.hh"
#include "io/EventReader.hh"
#include "io/ImportData.hh"
#include "io/EventReader.hh"
#include "io/RootImporter.hh"
#include "physics/base/CutoffParams.hh"
#include "physics/base/ModelData.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/base/ImportedProcessAdapter.hh"
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

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Check that volume names are consistent between the ROOT file and geometry
bool volumes_are_consistent(const GeoParams&                 geo,
                            const std::vector<ImportVolume>& rootinp)
{
    return geo.num_volumes() == rootinp.size()
           && std::all_of(RangeIter<VolumeId>(VolumeId{0}),
                          RangeIter<VolumeId>(VolumeId{geo.num_volumes()}),
                          [&](VolumeId vol) {
                              return geo.id_to_label(vol)
                                     == rootinp[vol.unchecked_get()].name;
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
                       {"storage_factor", v.storage_factor},
                       {"secondary_stack_factor", v.secondary_stack_factor},
                       {"use_device", v.use_device}};
}

void from_json(const nlohmann::json& j, LDemoArgs& v)
{
    j.at("geometry_filename").get_to(v.geometry_filename);
    j.at("physics_filename").get_to(v.physics_filename);
    j.at("hepmc3_filename").get_to(v.hepmc3_filename);
    j.at("seed").get_to(v.seed);
    j.at("max_num_tracks").get_to(v.max_num_tracks);
    j.at("max_steps").get_to(v.max_steps);
    j.at("storage_factor").get_to(v.storage_factor);
    j.at("secondary_stack_factor").get_to(v.secondary_stack_factor);
    j.at("use_device").get_to(v.use_device);
}

//---------------------------------------------------------------------------//
TransporterInput load_input(const LDemoArgs& args)
{
    CELER_LOG(status) << "Loading input files";
    TransporterInput result;

    // Load rootinp from ROOT file
    auto rootinp = RootImporter(args.physics_filename.c_str())();

    // Load geometry
    {
        result.geometry
            = std::make_shared<GeoParams>(args.geometry_filename.c_str());
    }

    // Load materials
    {
        result.materials = MaterialParams::from_import(rootinp);
    }

    // Create geometry/material coupling
    {
        GeoMaterialParams::Input input;
        input.geometry  = result.geometry;
        input.materials = result.materials;

        input.volume_to_mat.resize(input.geometry->num_volumes());
        for (auto volume_idx : range(rootinp.volumes.size()))
        {
            input.volume_to_mat[volume_idx]
                = MaterialId{rootinp.volumes[volume_idx].material_id};
        }
        if (!volumes_are_consistent(*input.geometry, rootinp.volumes))
        {
            // Volume names do not match exactly between exported ROOT file and
            // the geometry (possibly using a ROOT/GDML input with an ORANGE
            // geometry): try to let the GeoMaterialParams remap them
            CELER_LOG(warning) << "Volume/material mapping is inconsistent "
                                  "between ROOT file and geometry file: "
                                  "attempting to remap";
            input.volume_names.resize(rootinp.volumes.size());
            for (auto volume_idx : range(rootinp.volumes.size()))
            {
                input.volume_names[volume_idx]
                    = std::move(rootinp.volumes[volume_idx].name);
            }
        }
        result.geo_mats = std::make_shared<GeoMaterialParams>(std::move(input));
    }

    // Construct particle params
    {
        result.particles = ParticleParams::from_import(rootinp);
    }

    // Construct cutoffs
    {
        result.cutoffs = CutoffParams::from_import(
            rootinp, result.particles, result.materials);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles = result.particles;
        input.materials = result.materials;

        BremsstrahlungProcess::Options brem_options;
        brem_options.combined_model = args.combined_brem;
        brem_options.enable_lpm     = args.enable_lpm;

        auto process_data = std::make_shared<ImportedProcesses>(
            std::move(rootinp.processes));
        input.processes.push_back(
            std::make_shared<ComptonProcess>(result.particles, process_data));
        input.processes.push_back(std::make_shared<PhotoelectricProcess>(
            result.particles, result.materials, process_data));
        input.processes.push_back(std::make_shared<RayleighProcess>(
            result.particles, result.materials, process_data));
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
    input.primaries      = read_all_events();
    input.storage_factor = args.storage_factor;
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

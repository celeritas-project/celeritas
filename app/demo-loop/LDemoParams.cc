//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoParams.cc
//---------------------------------------------------------------------------//
#include "LDemoParams.hh"

#include "comm/Logger.hh"
#include "io/EventReader.hh"
#include "io/ImportData.hh"
#include "io/EventReader.hh"
#include "io/RootImporter.hh"
#include "physics/base/ImportedProcessAdapter.hh"
#include "physics/em/BremsstrahlungProcess.hh"
#include "physics/em/ComptonProcess.hh"
#include "physics/em/EIonizationProcess.hh"
#include "physics/em/EPlusAnnihilationProcess.hh"
#include "physics/em/GammaConversionProcess.hh"
#include "physics/em/PhotoelectricProcess.hh"
#include "physics/em/RayleighProcess.hh"
#include "LDemoIO.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
LDemoParams load_params(const LDemoArgs& args)
{
    CELER_LOG(status) << "Loading input files";
    LDemoParams result;

    // Load data from ROOT file
    const auto data = RootImporter(args.physics_filename.c_str())();

    // Load geometry
    {
        result.geometry
            = std::make_shared<GeoParams>(args.geometry_filename.c_str());
    }

    // Load materials
    {
        result.materials = MaterialParams::from_import(data);
    }

    // Create geometry/material coupling
    {
        GeoMaterialParams::Input input;
        input.geometry  = result.geometry;
        input.materials = result.materials;

        input.volume_to_mat
            = std::vector<MaterialId>(input.geometry->num_volumes());

        CELER_ASSERT(input.volume_to_mat.size() == data.volumes.size());

        for (const auto volume_id : range(data.volumes.size()))
        {
            const auto& volume             = data.volumes.at(volume_id);
            input.volume_to_mat[volume_id] = MaterialId{volume.material_id};
        }
        result.geo_mats = std::make_shared<GeoMaterialParams>(std::move(input));
    }

    // Construct particle params
    {
        result.particles = ParticleParams::from_import(data);
    }

    // Construct cutoffs
    {
        result.cutoffs = CutoffParams::from_import(
            data, result.particles, result.materials);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles = result.particles;
        input.materials = result.materials;

        auto process_data
            = std::make_shared<ImportedProcesses>(std::move(data.processes));
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
            result.particles, result.materials, process_data));

        result.physics = std::make_shared<PhysicsParams>(std::move(input));
    }

    // Load track initialization data
    {
        EventReader read_event(args.hepmc3_filename.c_str(), result.particles);
        TrackInitParams::Input input;
        input.primaries      = read_event();
        input.storage_factor = args.storage_factor;
        result.track_inits   = std::make_shared<TrackInitParams>(input);
    }

    // Construct RNG params
    {
        result.rng = std::make_shared<RngParams>(args.seed);
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_loop

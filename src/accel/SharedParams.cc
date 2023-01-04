//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.cc
//---------------------------------------------------------------------------//
#include "SharedParams.hh"

#include <CLHEP/Random/Random.h>
#include <G4TransportationManager.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputManager.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/StepCollector.hh"

#include "AlongStepFactory.hh"
#include "SetupOptions.hh"
#include "detail/HitManager.hh"

#if CELERITAS_USE_JSON
#    include "corecel/io/BuildOutput.hh"
#    include "corecel/sys/DeviceIO.json.hh"
#    include "corecel/sys/Environment.hh"
#    include "corecel/sys/EnvironmentIO.json.hh"
#    include "corecel/sys/KernelRegistry.hh"
#    include "corecel/sys/KernelRegistryIO.json.hh"
#    include "celeritas/global/ActionRegistryOutput.hh"
#    include "celeritas/phys/PhysicsParamsOutput.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Default destructor
SharedParams::~SharedParams() = default;

//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas using Geant4 data.
 *
 * This is a separate step from construction because it has to happen at the
 * beginning of the run, not when user classes are created. It should be called
 * from the "master" thread (for MT mode) or from the main thread (for Serial),
 * and it must complete before any worker thread tries to access the shared
 * data.
 */
SharedParams::SharedParams(const SetupOptions& options)
{
    CELER_EXPECT(!*this);

    SharedParams::InitializeWorker(options);

    CELER_LOG_LOCAL(status) << "Initializing Celeritas shared data";
    ScopedTimeLog scoped_time;

    this->initialize_impl(options);
    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize thread-local data.
 *
 * This is a separate step from construction because it has to happen at the
 * beginning of the run, not when user classes are created. It should be called
 * from all threads to ensure that construction is complete locally.
 */
void SharedParams::InitializeWorker(const SetupOptions& options)
{
    CELER_LOG_LOCAL(status) << "Initializing thread-local global data";
    ScopedTimeLog scoped_time;

    if (Device::num_devices() > 0)
    {
        // Initialize CUDA (you'll need to use CUDA environment variables to
        // control the preferred device)
        celeritas::activate_device(Device{0});

        // Heap size must be set before creating VecGeom device instance; and
        // let's just set the stack size as well
        if (options.cuda_stack_size > 0)
        {
            celeritas::set_cuda_stack_size(options.cuda_stack_size);
        }
        if (options.cuda_heap_size > 0)
        {
            celeritas::set_cuda_heap_size(options.cuda_heap_size);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear shared data after writing out diagnostics.
 *
 * This must be executed exactly *once* across all threads and at the end of
 * the run.
 */
void SharedParams::Finalize()
{
    CELER_EXPECT(*this);

    if (!output_filename_.empty())
    {
#if CELERITAS_USE_JSON
        CELER_LOG(info) << "Writing Celeritas output to \"" << output_filename_
                        << '"';
        OutputManager output;

        // System diagnostics
        output.insert(OutputInterfaceAdapter<Device>::from_const_ref(
            OutputInterface::Category::system, "device", celeritas::device()));
        output.insert(OutputInterfaceAdapter<KernelRegistry>::from_const_ref(
            OutputInterface::Category::system,
            "kernels",
            celeritas::kernel_registry()));
        output.insert(OutputInterfaceAdapter<Environment>::from_const_ref(
            OutputInterface::Category::system,
            "environ",
            celeritas::environment()));
        output.insert(std::make_shared<BuildOutput>());

        // Problem diagnostics
        output.insert(
            std::make_shared<PhysicsParamsOutput>(params_->physics()));
        output.insert(
            std::make_shared<ActionRegistryOutput>(params_->action_reg()));

        std::ofstream outf(output_filename_);
        CELER_VALIDATE(outf,
                       << "failed to open output file at \""
                       << output_filename_ << '"');
        output.output(&outf);
#else
        CELER_LOG(warning) << "JSON support is not enabled, so no output will "
                              "be written to \""
                           << output_filename_ << '"';
#endif
    }

    // Reset all data
    CELER_LOG_LOCAL(debug) << "Resetting shared parameters";
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from setup options.
 *
 * This is not thread-safe and should
 */
void SharedParams::initialize_impl(const SetupOptions& options)
{
    CELER_VALIDATE(options.make_along_step,
                   << "along-step action factory 'make_along_step' was not "
                      "defined in the celeritas::SetupOptions");

    celeritas::GeantImporter load_geant_data(GeantImporter::get_world_volume());
    auto imported = std::make_shared<ImportData>(load_geant_data());
    CELER_ASSERT(imported && *imported);

    CoreParams::Input params;

    // Create action manager
    {
        params.action_reg = std::make_shared<ActionRegistry>();
    }

    // Reload geometry
    if (!options.geometry_file.empty())
    {
        // Read directly from GDML input
        params.geometry = std::make_shared<GeoParams>(options.geometry_file);
    }
    else
    {
        // Import from Geant4
        params.geometry
            = std::make_shared<GeoParams>(GeantImporter::get_world_volume());
    }

    // Load materials
    {
        params.material = MaterialParams::from_import(*imported);
    }

    // Create geometry/material coupling
    {
        params.geomaterial = GeoMaterialParams::from_import(
            *imported, params.geometry, params.material);
    }

    // Construct particle params
    {
        params.particle = ParticleParams::from_import(*imported);
    }

    // Construct cutoffs
    {
        params.cutoff = CutoffParams::from_import(
            *imported, params.particle, params.material);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles       = params.particle;
        input.materials       = params.material;
        input.action_registry = params.action_reg.get();

        input.options.linear_loss_limit = imported->em_params.linear_loss_limit;
        input.options.secondary_stack_factor = options.secondary_stack_factor;

        {
            ProcessBuilder::Options opts;
            ProcessBuilder          build_process(
                *imported, opts, params.particle, params.material);

            std::set<ImportProcessClass> all_process_classes;
            for (const auto& p : imported->processes)
            {
                all_process_classes.insert(p.process_class);
            }
            for (auto p : all_process_classes)
            {
                input.processes.push_back(build_process(p));
            }
        }

        params.physics = std::make_shared<PhysicsParams>(std::move(input));
    }

    // Construct along-step action
    {
        AlongStepFactoryInput asfi;
        asfi.action_id = params.action_reg->next_id();
        asfi.geometry = params.geometry;
        asfi.material = params.material;
        asfi.geomaterial = params.geomaterial;
        asfi.particle = params.particle;
        asfi.cutoff = params.cutoff;
        asfi.physics = params.physics;
        asfi.imported = imported;

        auto along_step = options.make_along_step(asfi);
        CELER_VALIDATE(along_step,
                       << "along-step factory returned a null pointer");
        params.action_reg->insert(std::move(along_step));
    }

    // Construct RNG params
    {
        params.rng
            = std::make_shared<RngParams>(CLHEP::HepRandom::getTheSeed());
    }

    // Construct track initialization params
    {
        TrackInitParams::Input input;
        input.capacity   = options.initializer_capacity;
        input.max_events = options.max_num_events;
        params.init      = std::make_shared<TrackInitParams>(input);
    }

    // Construct sensitive detector callback
    if (options.sd)
    {
        hit_manager_ = std::make_shared<detail::HitManager>(*params.geometry,
                                                            options.sd);
        step_collector_ = std::make_shared<StepCollector>(
            StepCollector::VecInterface{hit_manager_},
            params.geometry,
            params.action_reg.get());
    }

    // Create params
    CELER_ASSERT(params);
    params_ = std::make_shared<CoreParams>(std::move(params));

    // Save other data as needed
    output_filename_ = options.output_file;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.cc
//---------------------------------------------------------------------------//
#include "SharedParams.hh"

#include <CLHEP/Random/Random.h>
#include <G4AutoLock.hh>
#include <G4TransportationManager.hh>

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
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "SetupOptions.hh"

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
 * Thread-safe setup of Celeritas using Geant4 data.
 *
 * This is a separate step from construction because it has to happen at the
 * beginning of the run, not when user classes are created. It should be called
 * from all threads to ensure that construction is complete locally.
 */
void SharedParams::Initialize(const SetupOptions& options)
{
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

    if (!*this)
    {
        // Maybe the first thread to run: build and store core params
        this->locked_initialize(options);
    }
    CELER_ENSURE(*this);
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
 * Construct from setup options in a thread-safe manner.
 */
void SharedParams::locked_initialize(const SetupOptions& options)
{
    static G4Mutex mutex = G4MUTEX_INITIALIZER;
    G4AutoLock     lock(&mutex);

    if (*this)
    {
        // Some other thread constructed params between the thread-unsafe check
        // and this thread-safe check
        return;
    }

    CELER_LOG_LOCAL(status) << "Initializing Celeritas";
    ScopedTimeLog scoped_time;

    celeritas::GeantImporter load_geant_data(GeantImporter::get_world_volume());
    auto imported = load_geant_data();
    CELER_ASSERT(imported);

    CoreParams::Input params;

    // Create action manager
    {
        params.action_reg = std::make_shared<ActionRegistry>();
    }

    // Load geometry
    {
        // TODO: export GDML through Geant4 to temporary file
        params.geometry = std::make_shared<GeoParams>(options.geometry_file);
    }

    // Load materials
    {
        params.material = MaterialParams::from_import(imported);
    }

    // Create geometry/material coupling
    {
        params.geomaterial = GeoMaterialParams::from_import(
            imported, params.geometry, params.material);
    }

    // Construct particle params
    {
        params.particle = ParticleParams::from_import(imported);
    }

    // Construct cutoffs
    {
        params.cutoff = CutoffParams::from_import(
            imported, params.particle, params.material);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles       = params.particle;
        input.materials       = params.material;
        input.action_registry = params.action_reg.get();

        input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
        input.options.secondary_stack_factor = options.secondary_stack_factor;

        {
            ProcessBuilder::Options opts;
            ProcessBuilder          build_process(
                imported, opts, params.particle, params.material);

            std::set<ImportProcessClass> all_process_classes;
            for (const auto& p : imported.processes)
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

    // TODO: different along-step kernels
    {
        // Create along-step action
        auto along_step = AlongStepGeneralLinearAction::from_params(
            params.action_reg->next_id(),
            *params.material,
            *params.particle,
            *params.physics,
            imported.em_params.energy_loss_fluct);
        params.action_reg->insert(along_step);
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
    {
        // TODO: interface for accepting other user hit callbacks
    }

    // Create params
    CELER_ASSERT(params);
    params_ = std::make_shared<CoreParams>(std::move(params));

    // Save other data as needed
    output_filename_ = options.output_file;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

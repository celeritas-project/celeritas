//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.cc
//---------------------------------------------------------------------------//
#include "SharedParams.hh"

#include <fstream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <CLHEP/Random/Random.h>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/io/EventWriter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/RootEventWriter.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Process.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/StepCollector.hh"

#include "AlongStepFactory.hh"
#include "SetupOptions.hh"
#include "detail/HitManager.hh"
#include "detail/OffloadWriter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
std::vector<std::shared_ptr<Process const>>
build_processes(ImportData const& imported,
                SetupOptions const& options,
                std::shared_ptr<ParticleParams const> const& particle,
                std::shared_ptr<MaterialParams const> const& material)
{
    // Build a list of processes to ignore
    ProcessBuilder::UserBuildMap ignore;
    for (std::string const& process_name : options.ignore_processes)
    {
        ImportProcessClass ipc;
        try
        {
            ipc = geant_name_to_import_process_class(process_name);
        }
        catch (RuntimeError const&)
        {
            CELER_LOG(warning) << "User-ignored process '" << process_name
                               << "' is unknown to Celeritas";
            continue;
        }
        ignore.emplace(ipc, WarnAndIgnoreProcess{ipc});
    }
    ProcessBuilder::Options opts;
    ProcessBuilder build_process(imported, particle, material, ignore, opts);

    // Build proceses
    std::vector<std::shared_ptr<Process const>> result;
    for (auto p : ProcessBuilder::get_all_process_classes(imported.processes))
    {
        result.push_back(build_process(p));
        if (!result.back())
        {
            // Deliberately ignored process
            CELER_LOG(debug) << "Ignored process class " << to_cstring(p);
            result.pop_back();
        }
    }

    CELER_VALIDATE(!result.empty(),
                   << "no supported physics processes were found");
    return result;
}

//---------------------------------------------------------------------------//
std::vector<G4ParticleDefinition const*>
build_g4_particles(std::shared_ptr<ParticleParams const> const& particles,
                   std::shared_ptr<PhysicsParams const> const& phys)
{
    CELER_EXPECT(particles);
    CELER_EXPECT(phys);

    G4ParticleTable* g4particles = G4ParticleTable::GetParticleTable();
    CELER_ASSERT(g4particles);

    std::vector<G4ParticleDefinition const*> result;

    for (auto par_id : range(ParticleId{particles->size()}))
    {
        if (phys->processes(par_id).empty())
        {
            CELER_LOG(warning)
                << "Not offloading particle '"
                << particles->id_to_label(par_id)
                << "' because it has no physics processes defined";
            continue;
        }

        PDGNumber pdg = particles->id_to_pdg(par_id);
        G4ParticleDefinition* g4pd = g4particles->FindParticle(pdg.get());
        CELER_VALIDATE(g4pd,
                       << "could not find PDG '" << pdg.get()
                       << "' in G4ParticleTable");
        result.push_back(g4pd);
    }

    CELER_ENSURE(!result.empty());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

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
SharedParams::SharedParams(SetupOptions const& options)
{
    CELER_EXPECT(!*this);

    CELER_LOG_LOCAL(status) << "Initializing Celeritas shared data";
    ScopedMem record_mem("SharedParams.construct");
    ScopedTimeLog scoped_time;

    // Initialize device and other "global" data
    SharedParams::initialize_device(options);

    // Construct core data
    this->initialize_core(options);

    // Set up output after params are constructed
    this->try_output();

    if (!options.offload_output_file.empty())
    {
        std::unique_ptr<EventWriterInterface> writer;
        if (ends_with(options.offload_output_file, ".root"))
        {
            writer.reset(
                new RootEventWriter(std::make_shared<RootFileManager>(
                                        options.offload_output_file.c_str()),
                                    params_->particle()));
        }
        else
        {
            writer.reset(new EventWriter(options.offload_output_file,
                                         params_->particle()));
        }
        offload_writer_
            = std::make_shared<detail::OffloadWriter>(std::move(writer));
    }

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * On worker threads, set up data with thread storage duration.
 *
 * Some data that has "static" storage duration (such as CUDA device
 * properties) in single-thread mode has "thread" storage in a multithreaded
 * application. It must be initialized on all threads.
 */
void SharedParams::InitializeWorker(SetupOptions const& options)
{
    CELER_LOG_LOCAL(status) << "Initializing worker thread";
    ScopedTimeLog scoped_time;
    return SharedParams::initialize_device(options);
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

    // Output at end of run
    this->try_output();

    // Reset all data
    CELER_LOG_LOCAL(debug) << "Resetting shared parameters";
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize GPU device on each thread.
 *
 * This is thread safe and must be called from every worker thread.
 */
void SharedParams::initialize_device(SetupOptions const& options)
{
    if (Device::num_devices() == 0)
    {
        // No GPU is enabled so no global initialization is needed
        return;
    }

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

//---------------------------------------------------------------------------//
/*!
 * Construct from setup options.
 *
 * This is not thread-safe and should only be called from a single CPU thread
 * that is guaranteed to complete the initialization before any other threads
 * try to access the shared data.
 */
void SharedParams::initialize_core(SetupOptions const& options)
{
    CELER_VALIDATE(options.make_along_step,
                   << "along-step action factory 'make_along_step' was not "
                      "defined in the celeritas::SetupOptions");

    auto const imported = [&options] {
        celeritas::GeantImporter load_geant_data(
            GeantImporter::get_world_volume());
        // Convert ImportVolume names to GDML versions if we're exporting
        GeantImportDataSelection import_opts;
        import_opts.particles = GeantImportDataSelection::em_basic;
        import_opts.processes = import_opts.particles;
        import_opts.unique_volumes = options.geometry_file.empty();
        return std::make_shared<ImportData>(load_geant_data(import_opts));
    }();
    CELER_ASSERT(imported && !imported->particles.empty()
                 && !imported->materials.empty()
                 && !imported->processes.empty() && !imported->volumes.empty());

    if (!options.physics_output_file.empty())
    {
        RootExporter export_root(options.physics_output_file.c_str());
        export_root(*imported);
    }

    CoreParams::Input params;

    // Create registries
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::make_shared<OutputRegistry>();

    // Load geometry
    params.geometry = [&options] {
        if (!options.geometry_file.empty())
        {
            // Read directly from GDML input
            return std::make_shared<GeoParams>(options.geometry_file);
        }
        else
        {
            // Import from Geant4
            return std::make_shared<GeoParams>(
                GeantImporter::get_world_volume());
        }
    }();

    // Load materials
    params.material = MaterialParams::from_import(*imported);

    // Create geometry/material coupling
    params.geomaterial = GeoMaterialParams::from_import(
        *imported, params.geometry, params.material);

    // Construct particle params
    params.particle = ParticleParams::from_import(*imported);

    // Construct cutoffs
    params.cutoff = CutoffParams::from_import(
        *imported, params.particle, params.material);

    // Load physics: create individual processes with make_shared
    params.physics = [&params, &options, &imported] {
        PhysicsParams::Input input;
        input.particles = params.particle;
        input.materials = params.material;
        input.processes = build_processes(
            *imported, options, params.particle, params.material);
        input.relaxation = nullptr;  // TODO: add later?
        input.action_registry = params.action_reg.get();

        input.options.linear_loss_limit = imported->em_params.linear_loss_limit;
        input.options.lowest_electron_energy = PhysicsParamsOptions::Energy{
            imported->em_params.lowest_electron_energy};
        input.options.secondary_stack_factor = options.secondary_stack_factor;

        return std::make_shared<PhysicsParams>(std::move(input));
    }();

    // Construct RNG params
    params.rng = std::make_shared<RngParams>(CLHEP::HepRandom::getTheSeed());

    // Construct simulation params
    params.sim = SimParams::from_import(*imported, params.particle);

    // Construct track initialization params
    params.init = [&options] {
        TrackInitParams::Input input;
        input.capacity = options.initializer_capacity;
        input.max_events = options.max_num_events;
        input.track_order = options.track_order;
        return std::make_shared<TrackInitParams>(std::move(input));
    }();

    if (options.get_num_streams)
    {
        int num_streams = options.get_num_streams();
        CELER_VALIDATE(num_streams > 0,
                       << "nonpositive number of streams (" << num_streams
                       << ") returned by SetupOptions.get_num_streams");
        params.max_streams = num_streams;
    }
    else
    {
        // Default to setting the maximum number of streams based on Geant4
        // multithreading.
        // Hit processors *must* be allocated on the thread they're used
        // because of geant4 thread-local SDs. There must be one per thread.
        params.max_streams = [] {
            auto* run_man = G4RunManager::GetRunManager();
            CELER_VALIDATE(run_man,
                           << "G4RunManager was not created before "
                              "initializing SharedParams");
            return celeritas::get_num_threads(*run_man);
        }();
    }

    // Allocate device streams, or use the default stream if there is only one.
    if (celeritas::device() && !options.default_stream
        && params.max_streams > 1)
    {
        celeritas::device().create_streams(params.max_streams);
    }

    // Construct along-step action
    params.action_reg->insert([&params, &options, &imported] {
        AlongStepFactoryInput asfi;
        asfi.action_id = params.action_reg->next_id();
        asfi.geometry = params.geometry;
        asfi.material = params.material;
        asfi.geomaterial = params.geomaterial;
        asfi.particle = params.particle;
        asfi.cutoff = params.cutoff;
        asfi.physics = params.physics;
        asfi.imported = imported;
        auto const along_step{options.make_along_step(asfi)};
        CELER_VALIDATE(along_step,
                       << "along-step factory returned a null pointer");
        return along_step;
    }());

    // Construct sensitive detector callback
    if (options.sd)
    {
        hit_manager_ = std::make_shared<detail::HitManager>(
            *params.geometry, *params.particle, options.sd, params.max_streams);
        step_collector_ = std::make_shared<StepCollector>(
            StepCollector::VecInterface{hit_manager_},
            params.geometry,
            params.max_streams,
            params.action_reg.get());
    }

    // Create params
    CELER_ASSERT(params);
    params_ = std::make_shared<CoreParams>(std::move(params));

    // Translate supported particles
    particles_ = build_g4_particles(params_->particle(), params_->physics());

    // Save output filename
    output_filename_ = options.output_file;
}

//---------------------------------------------------------------------------//
/*!
 * Write available Celeritas output.
 *
 * This can be done multiple times, overwriting the same file so that we can
 * get output before construction and after
 */
void SharedParams::try_output() const
{
    CELER_EXPECT(params_);
    if (output_filename_.empty())
    {
        return;
    }

#if CELERITAS_USE_JSON
    CELER_LOG(info) << "Writing Celeritas output to \"" << output_filename_
                    << '"';

    std::ofstream outf(output_filename_);
    CELER_VALIDATE(
        outf, << "failed to open output file at \"" << output_filename_ << '"');
    params_->output_reg()->output(&outf);
#else
    CELER_LOG(warning)
        << "JSON support is not enabled, so no output will be written to \""
        << output_filename_ << '"';
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

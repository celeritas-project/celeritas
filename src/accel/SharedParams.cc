//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SharedParams.cc
//---------------------------------------------------------------------------//
#include "SharedParams.hh"

#include <fstream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>
#include <CLHEP/Random/Random.h>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4Positron.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4VisExtent.hh>

#include "corecel/Config.hh"
#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/MemRegistry.hh"
#include "corecel/sys/MemRegistryIO.json.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/g4/GeantGeoParams.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSd.hh"
#include "celeritas/ext/GeantSdOutput.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Scoring.hh"
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
#include "celeritas/user/SlotDiagnostic.hh"
#include "celeritas/user/StepCollector.hh"

#include "AlongStepFactory.hh"
#include "SetupOptions.hh"

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
std::vector<G4ParticleDefinition*>
build_g4_particles(std::shared_ptr<ParticleParams const> const& particles,
                   std::shared_ptr<PhysicsParams const> const& phys)
{
    CELER_EXPECT(particles);
    CELER_EXPECT(phys);

    G4ParticleTable* g4particles = G4ParticleTable::GetParticleTable();
    CELER_ASSERT(g4particles);

    std::vector<G4ParticleDefinition*> result;

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
/*!
 * Shared static mutex for once-only updated parameters.
 */
std::mutex& updating_mutex()
{
    static std::mutex result;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether celeritas is disabled, set to kill, or to be enabled.
 *
 * This gets the value from environment variables and
 *
 * \todo This will be refactored for 0.6 to take a \c celeritas::inp object and
 * determine values rather than from the environment .
 */
auto SharedParams::GetMode() -> Mode
{
    using Mode = SharedParams::Mode;

    static bool const kill_offload = [] {
        if (celeritas::getenv("CELER_KILL_OFFLOAD").empty())
            return false;

        CELER_LOG(info) << "Killing Geant4 tracks supported by Celeritas "
                           "offloading since the 'CELER_KILL_OFFLOAD' "
                        << "environment variable is present and non-empty";
        return true;
    }();
    static bool const disabled = [] {
        if (celeritas::getenv("CELER_DISABLE").empty())
            return false;

        if (kill_offload)
        {
            CELER_LOG(warning)
                << "DEPRECATED (remove in 0.7): both CELER_DISABLE "
                   "and CELER_KILL_OFFLOAD environment variables "
                   "were defined: choose one";
            return false;
        }

        CELER_LOG(info)
            << "Disabling Celeritas offloading since the 'CELER_DISABLE' "
            << "environment variable is present and non-empty";
        return true;
    }();

    if (disabled)
    {
        return Mode::disabled;
    }
    else if (kill_offload)
    {
        return Mode::kill_offload;
    }
    return Mode::enabled;
}

//---------------------------------------------------------------------------//
bool SharedParams::CeleritasDisabled()
{
    return GetMode() == Mode::disabled;
}

//---------------------------------------------------------------------------//
bool SharedParams::KillOffloadTracks()
{
    return GetMode() == Mode::kill_offload;
}

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

    ScopedProfiling profile_this{"construct-params"};
    ScopedMem record_mem("SharedParams.construct");
    ScopedTimeLog scoped_time;

    mode_ = GetMode();
    if (mode_ == Mode::kill_offload)
    {
        // In a Geant4-only simulation, use a hardcoded list
        particles_ = {
            G4Gamma::Gamma(),
            G4Electron::Electron(),
            G4Positron::Positron(),
        };
    }

    if (mode_ != Mode::enabled)
    {
        // Stop initializing but create output registry for diagnostics
        output_reg_ = std::make_shared<OutputRegistry>();
        output_filename_ = options.output_file;

        if (!output_filename_.empty())
        {
            CELER_LOG(debug)
                << R"(Constructing output registry for no-offload run)";

            // Celeritas core params didn't add system metadata: do it
            // ourselves to save system diagnostic information
            output_reg_->insert(
                OutputInterfaceAdapter<MemRegistry>::from_const_ref(
                    OutputInterface::Category::system,
                    "memory",
                    celeritas::mem_registry()));
            output_reg_->insert(
                OutputInterfaceAdapter<Environment>::from_const_ref(
                    OutputInterface::Category::system,
                    "environ",
                    celeritas::environment()));
            output_reg_->insert(std::make_shared<BuildOutput>());
        }

        return;
    }

    CELER_LOG(info) << "Activating Celeritas version " << version_string
                    << " on " << (Device::num_devices() > 0 ? "GPU" : "CPU");

    // Initialize CUDA (CUDA environment variables control the preferred
    // device)
    celeritas::activate_device();

    if (celeritas::device() && CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
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

    // Construct core data
    this->initialize_core(options);
    CELER_ASSERT(params_ && output_reg_);

    // Translate supported particles
    particles_ = build_g4_particles(params_->particle(), params_->physics());

    // Create bounding box from navigator geometry
    bbox_ = [] {
        G4VPhysicalVolume const* world = GeantImporter::get_world_volume();
        CELER_ASSERT(world);
        G4LogicalVolume const* world_lv = world->GetLogicalVolume();
        CELER_ASSERT(world_lv);
        G4VSolid const* solid = world_lv->GetSolid();
        CELER_ASSERT(solid);
        G4VisExtent const& extent = solid->GetExtent();

        BBox result{{extent.GetXmin(), extent.GetYmin(), extent.GetZmin()},
                    {extent.GetXmax(), extent.GetYmax(), extent.GetZmax()}};
        CELER_VALIDATE(result,
                       << "world bounding box {" << result.lower() << ", "
                       << result.upper() << "} is invalid");
        return result;
    }();

    // Save output filename (possibly empty if disabling output)
    output_filename_ = options.output_file;

    if (output_filename_ != "-")
    {
        // Write output after params are constructed before anything can go
        // wrong
        this->try_output();
    }
    else
    {
        CELER_LOG(debug)
            << R"(Skipping 'startup' JSON output since writing to stdout)";
    }

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
void SharedParams::InitializeWorker(SetupOptions const&)
{
    CELER_EXPECT(*this);

    celeritas::activate_device_local();
}

//---------------------------------------------------------------------------//
/*!
 * Clear shared data after writing out diagnostics.
 *
 * This should be executed exactly *once* across all threads and at the end of
 * the run.
 */
void SharedParams::Finalize()
{
    static std::mutex finalize_mutex;
    std::lock_guard scoped_lock{finalize_mutex};

    // Output at end of run
    this->try_output();

    // Reset all data
    CELER_LOG_LOCAL(debug) << "Resetting shared parameters";
    *this = {};

    // Reset streams before the static destructor does
    celeritas::device().destroy_streams();

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Let LocalTransporter register the thread's state.
 */
void SharedParams::set_state(unsigned int stream_id, SPState&& state)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(stream_id < states_.size());
    CELER_EXPECT(state);
    CELER_EXPECT(!states_[stream_id]);

    states_[stream_id] = std::move(state);
}

//---------------------------------------------------------------------------//
/*!
 * Lazily obtained number of streams.
 */
unsigned int SharedParams::num_streams() const
{
    if (CELER_UNLIKELY(states_.empty()))
    {
        // Initial lock-free check failed; now lock and create if needed
        // Default to setting the maximum number of streams based on Geant4
        // run manager.
        const_cast<SharedParams*>(this)->set_num_streams(
            celeritas::get_geant_num_threads());
    }

    CELER_ENSURE(!states_.empty());
    return states_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Lazily created Geant geometry parameters.
 */
auto SharedParams::geant_geo_params() const -> SPConstGeantGeoParams const&
{
    if (CELER_UNLIKELY(!geant_geo_))
    {
        // Initial lock-free check failed; now lock and create if needed
        std::lock_guard scoped_lock{updating_mutex()};
        if (!geant_geo_)
        {
            CELER_LOG_LOCAL(debug) << "Constructing GeantGeoParams wrapper";

            auto geo_params = std::make_shared<GeantGeoParams>(
                GeantImporter::get_world_volume());
            const_cast<SharedParams*>(this)->geant_geo_ = std::move(geo_params);
            CELER_ENSURE(geant_geo_);
        }
    }
    return geant_geo_;
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
        // TODO: optical particle/process import
        GeantImportDataSelection import_opts;
        import_opts.particles = GeantImportDataSelection::em_basic;
        import_opts.processes = import_opts.particles;
        import_opts.unique_volumes = options.geometry_file.empty();
        return std::make_shared<ImportData>(load_geant_data(import_opts));
    }();
    CELER_ASSERT(imported && !imported->particles.empty()
                 && !imported->geo_materials.empty()
                 && !imported->phys_materials.empty()
                 && !imported->processes.empty() && !imported->volumes.empty());

    if (!options.physics_output_file.empty())
    {
        RootExporter export_root(options.physics_output_file.c_str());
        export_root(*imported);
    }

    if (!options.geometry_output_file.empty())
    {
        CELER_VALIDATE(options.geometry_file.empty(),
                       << "the 'geometry_output_file' option cannot be used "
                          "when manually loading a geometry (the "
                          "'geometry_file' option is also set)");

        save_gdml(GeantImporter::get_world_volume(),
                  options.geometry_output_file);
    }

    CoreParams::Input params;

    // Create registries
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::make_shared<OutputRegistry>();
    output_reg_ = params.output_reg;

    // Load geometry
    params.geometry = [&options] {
        if (!options.geometry_file.empty())
        {
            // Read directly from GDML input
            return std::make_shared<GeoParams>(options.geometry_file);
        }
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT
        else if (geant_geo_)
        {
            // Lazily created geo was requested first
            return geant_geo_;
        }
#endif
        else
        {
            // Import from Geant4
            return std::make_shared<GeoParams>(
                GeantImporter::get_world_volume());
        }
    }();

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT
    // Save the Geant4 geometry since we've already made it
    geant_geo_ = params.geometry;
#endif

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

    // Construct shared data for Coulomb scattering
    params.wentzel = WentzelOKVIParams::from_import(
        *imported, params.material, params.particle);

    // Load physics: create individual processes with make_shared
    params.physics = [&params, &options, &imported] {
        PhysicsParams::Input input;
        input.particles = params.particle;
        input.materials = params.material;
        input.processes = build_processes(
            *imported, options, params.particle, params.material);
        input.relaxation = nullptr;  // TODO: add later?
        input.action_registry = params.action_reg.get();

        // Set physics options
        input.options.linear_loss_limit = imported->em_params.linear_loss_limit;
        input.options.light.lowest_energy = ParticleOptions::Energy(
            imported->em_params.lowest_electron_energy);
        input.options.heavy.lowest_energy
            = ParticleOptions::Energy(imported->em_params.lowest_muhad_energy);
        input.options.secondary_stack_factor = options.secondary_stack_factor;

        // Set multiple scattering options
        input.options.light.range_factor = imported->em_params.msc_range_factor;
        input.options.heavy.range_factor
            = imported->em_params.msc_muhad_range_factor;
        input.options.safety_factor = imported->em_params.msc_safety_factor;
        input.options.lambda_limit = imported->em_params.msc_lambda_limit;
        input.options.light.displaced = imported->em_params.msc_displaced;
        input.options.heavy.displaced = imported->em_params.msc_muhad_displaced;
        input.options.light.step_limit_algorithm
            = imported->em_params.msc_step_algorithm;
        input.options.heavy.step_limit_algorithm
            = imported->em_params.msc_muhad_step_algorithm;

        return std::make_shared<PhysicsParams>(std::move(input));
    }();

    // Construct RNG params
    params.rng = std::make_shared<RngParams>(CLHEP::HepRandom::getTheSeed());

    // Construct simulation params
    params.sim = std::make_shared<SimParams>([&] {
        auto input = SimParams::Input::from_import(
            *imported, params.particle, options.max_field_substeps);
        if (options.max_steps != SetupOptions::no_max_steps())
        {
            input.max_steps = options.max_steps;
        }
        return input;
    }());

    if (options.max_num_events > 0)
    {
        CELER_LOG(warning) << "Deprecated option 'max_events': will be "
                              "removed in v0.6";
    }

    // Construct track initialization params
    params.init = [&options] {
        TrackInitParams::Input input;
        input.capacity = options.initializer_capacity;
        input.max_events = 1;  // TODO: use special "max events" case
        input.track_order = options.track_order;
        if (input.track_order == TrackOrder::size_)
        {
            input.track_order = celeritas::device() ? TrackOrder::init_charge
                                                    : TrackOrder::none;
            CELER_LOG(info) << "Set track ordering to default: "
                            << to_cstring(input.track_order);
        }
        return std::make_shared<TrackInitParams>(std::move(input));
    }();

    if (options.get_num_streams)
    {
        int num_streams = options.get_num_streams();
        CELER_VALIDATE(num_streams > 0,
                       << "nonpositive number of streams (" << num_streams
                       << ") returned by SetupOptions.get_num_streams");
        params.max_streams = static_cast<size_type>(num_streams);
        this->set_num_streams(num_streams);
        CELER_ASSERT(this->num_streams()
                     == static_cast<unsigned int>(num_streams));
    }
    else
    {
        // Default to setting the maximum number of streams based on Geant4
        // multithreading.
        params.max_streams = this->num_streams();
    }

    // Set state size
    params.tracks_per_stream = options.max_num_tracks;

    // Allocate device streams
    if (auto& d = celeritas::device())
    {
        d.create_streams(params.max_streams);
    }
    if (options.default_stream)
    {
        CELER_LOG(warning) << "Ignoring removed option 'default_stream'";
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
        auto along_step{options.make_along_step(asfi)};
        CELER_VALIDATE(along_step,
                       << "along-step factory returned a null pointer");
        return along_step;
    }());

    // Create params
    CELER_ASSERT(params);
    params_ = std::make_shared<CoreParams>(std::move(params));

    // Construct sensitive detector callback
    if (options.sd)
    {
        hit_manager_ = std::make_shared<GeantSd>(params_->geometry(),
                                                 *params_->particle(),
                                                 to_inp(options.sd),
                                                 params_->max_streams());
        step_collector_
            = StepCollector::make_and_insert(*params_, {hit_manager_});
        output_reg_->insert(std::make_shared<GeantSdOutput>(hit_manager_));
    }

    // Add diagnostics
    if (!options.slot_diagnostic_prefix.empty())
    {
        SlotDiagnostic::make_and_insert(*params_,
                                        options.slot_diagnostic_prefix);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Save the number of streams (thread-safe).
 *
 * This could be obtained from the run manager *or* set by the user.
 */
void SharedParams::set_num_streams(unsigned int num_streams)
{
    CELER_EXPECT(num_streams > 0);

    std::lock_guard scoped_lock{updating_mutex()};
    if (!states_.empty() && states_.size() != num_streams)
    {
        // This could happen if someone queries the number of streams
        // before initializing celeritas
        CELER_LOG(warning) << "Changing number of streams from "
                           << states_.size() << " to user-specified "
                           << num_streams;
    }
    else
    {
        CELER_LOG_LOCAL(debug)
            << "Setting number of streams to " << num_streams;
    }

    states_.resize(num_streams);
}

//---------------------------------------------------------------------------//
/*!
 * Write available Celeritas output.
 *
 * This can be done multiple times, overwriting the same file so that we can
 * get output before construction *and* after.
 */
void SharedParams::try_output() const
{
    std::string filename = output_filename_;
    if (filename.empty())
    {
        CELER_LOG(debug) << "Skipping output: SetupOptions::output_file is "
                            "empty";
        return;
    }

    auto msg = CELER_LOG(info);
    msg << "Wrote Geant4 diagnostic output to ";
    std::ofstream outf;
    std::ostream* os{nullptr};
    if (filename == "-")
    {
        os = &std::cout;
        msg << "<stdout>";
    }
    else
    {
        os = &outf;
        outf.open(filename);
        CELER_VALIDATE(
            outf, << "failed to open output file at \"" << filename << '"');
        msg << '"' << filename << '"';
    }
    CELER_ASSERT(os);
    output_reg_->output(os);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

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
#include <G4MuonMinus.hh>
#include <G4MuonPlus.hh>
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
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/random/params/RngParams.hh"
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
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/ext/GeantSd.hh"
#include "celeritas/ext/GeantSdOutput.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/FrameworkInput.hh"
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
#include "celeritas/setup/FrameworkInput.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/SlotDiagnostic.hh"
#include "celeritas/user/StepCollector.hh"

#include "AlongStepFactory.hh"
#include "SetupOptions.hh"
#include "TimeOutput.hh"

#include "detail/IntegrationSingleton.hh"
#include "detail/OffloadWriter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
void verify_offload(std::vector<G4ParticleDefinition*> const& offload,
                    ParticleParams const& particles,
                    PhysicsParams const& phys)
{
    std::vector<bool> found_particle(particles.size(), false);
    std::vector<G4ParticleDefinition const*> missing;

    for (auto const* pd : offload)
    {
        ParticleId pid;
        if (pd)
        {
            PDGNumber pdg{pd->GetPDGEncoding()};
            CELER_VALIDATE(
                pdg, << "unsupported particle type: " << PrintablePD{pd});
            pid = particles.find(pdg);
        }
        if (pid)
        {
            found_particle[pid.get()] = true;
            if (phys.processes(pid).empty())
            {
                CELER_LOG(warning) << "User-selected offload particle '"
                                   << particles.id_to_label(pid)
                                   << "' has no physics processes defined";
            }
        }
        else
        {
            missing.push_back(pd);
        }
    }

    auto printable_pd
        = [](G4ParticleDefinition const* p) { return PrintablePD{p}; };
    CELER_VALIDATE(missing.empty(),
                   << "not all particles from TrackingManagerConstructor are "
                      "active in Celeritas: missing "
                   << join(missing.begin(), missing.end(), ", ", printable_pd));

    if (found_particle != std::vector<bool>(particles.size(), true))
    {
        //! \todo Overhaul DataSelection Flags and GeantImporter
        CELER_LOG(warning)
            << "Mismatch between ParticlesParams (size " << particles.size()
            << ") and user-defined offload list (size " << offload.size()
            << "). Geant4 data import is not properly defined.";
    }
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
/*!
 * Get a list of all supported particles.
 */
auto SharedParams::supported_offload_particles() -> VecG4PD const&
{
    static VecG4PD const supported_particles = {
        G4Electron::Definition(),
        G4Positron::Definition(),
        G4Gamma::Definition(),
        G4MuonMinus::Definition(),
        G4MuonPlus::Definition(),
    };

    return supported_particles;
}

//---------------------------------------------------------------------------//
/*!
 * Get the list of default particles offloaded in Geant4 applications.
 *
 * If no user-defined list is provided, this defaults to simulating EM showers.
 */
auto SharedParams::default_offload_particles() -> VecG4PD const&
{
    static VecG4PD const default_particles = {
        G4Electron::Definition(),
        G4Positron::Definition(),
        G4Gamma::Definition(),
    };

    return default_particles;
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

    if (mode_ == Mode::enabled || mode_ == Mode::kill_offload)
    {
        // Set up offloaded particles based on user input
        auto const& user_offload = options.offload_particles;
        offload_particles_ = user_offload.empty() ? default_offload_particles()
                                                  : user_offload;
    }

    if (mode_ != Mode::enabled)
    {
        // Stop initializing but create output registry for diagnostics
        output_reg_ = std::make_shared<OutputRegistry>();
        output_filename_ = options.output_file;

        // Create the timing output
        timer_
            = std::make_shared<TimeOutput>(celeritas::get_geant_num_threads());

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
            output_reg_->insert(timer_);
        }

        return;
    }

    // Construct input and then build the problem setup
    auto framework_inp = to_inp(options);
    auto loaded = setup::framework_input(framework_inp);
    params_ = std::move(loaded.problem.core_params);
    optical_ = std::move(loaded.problem.optical_collector);
    output_filename_ = loaded.problem.output_file;
    CELER_ASSERT(params_);

    // Load geant4 geometry adapter and save as "global"
    CELER_ASSERT(loaded.geo);
    geant_geo_ = std::move(loaded.geo);
    celeritas::global_geant_geo(geant_geo_);

    // Save built attributes
    output_reg_ = params_->output_reg();
    geant_sd_ = std::move(loaded.problem.geant_sd);
    step_collector_ = std::move(loaded.problem.step_collector);

    // Translate supported particles
    verify_offload(
        offload_particles_, *params_->particle(), *params_->physics());

    // Create bounding box from navigator geometry
    bbox_ = geant_geo_->get_clhep_bbox();

    // Create streams
    this->set_num_streams(params_->max_streams());

    // Add timing output
    timer_ = std::make_shared<TimeOutput>(params_->max_streams());
    output_reg_->insert(timer_);

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

    if (auto const& offload_file = loaded.problem.offload_file;
        !offload_file.empty())
    {
        std::unique_ptr<EventWriterInterface> writer;
        if (ends_with(offload_file, ".root"))
        {
            writer.reset(new RootEventWriter(
                std::make_shared<RootFileManager>(offload_file.c_str()),
                params_->particle()));
        }
        else
        {
            writer.reset(new EventWriter(offload_file, params_->particle()));
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
    CELER_LOG(debug) << "Resetting shared parameters";
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
    CELER_EXPECT(!states_.empty());

    CELER_EXPECT(stream_id < states_.size());
    CELER_EXPECT(state);
    CELER_EXPECT(!states_[stream_id]);

    states_[stream_id] = std::move(state);
}

//---------------------------------------------------------------------------//
/*!
 * Lazily obtained number of streams.
 *
 * \todo This is currently needed due to some wackiness with the
 * celer-g4 GeantDiagnostics and initialization sequence. We should remove
 * this: for user applications it should strictly be determined by the run
 * manager/input, and celer-g4 should set it up accordingly.
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
        CELER_LOG(debug) << "Setting number of streams to " << num_streams;
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

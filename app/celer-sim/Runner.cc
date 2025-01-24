//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/Runner.cc
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include <functional>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/global/CoreParams.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/io/RootCoreParamsOutput.hh"
#include "celeritas/io/RootEventReader.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/CherenkovParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/optical/ScintillationParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/phys/PrimaryGenerator.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/phys/Process.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/phys/RootEventSampler.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/ActionDiagnostic.hh"
#include "celeritas/user/RootStepWriter.hh"
#include "celeritas/user/SimpleCalo.hh"
#include "celeritas/user/SlotDiagnostic.hh"
#include "celeritas/user/StepCollector.hh"
#include "celeritas/user/StepData.hh"
#include "celeritas/user/StepDiagnostic.hh"

#include "RunnerInput.hh"
#include "Transporter.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
Runner::Runner(RunnerInput const& inp)
{
    CELER_VALIDATE(inp.event_file.empty() != !inp.primary_options,
                   << "either a event filename or options to generate "
                      "primaries must be provided (but not both)");
    CELER_VALIDATE(!inp.mctruth_filter || !inp.mctruth_file.empty(),
                   << "'mctruth_filter' cannot be specified without providing "
                      "'mctruth_file'");

    using SPImporter = std::shared_ptr<ImporterInterface>;

    // Possible Geant4 world volume so we can reuse geometry
    G4VPhysicalVolume const* g4world{nullptr};

    // Import data and load geometry
    // If Geant4 is initialized, its data is scoped by the GeantImporter
    auto import = [&inp, &g4world]() -> SPImporter {
        if (ends_with(inp.physics_file, ".root"))
        {
            // Load from ROOT file
            return std::make_shared<RootImporter>(inp.physics_file);
        }

        std::string const& filename
            = !inp.physics_file.empty() ? inp.physics_file : inp.geometry_file;

        // Load Geant4 and retain to use geometry
        GeantSetup setup(filename, inp.physics_options);
        g4world = setup.world();
        return std::make_shared<GeantImporter>(std::move(setup));
    }();

    // Import physics
    auto const imported = (*import)();

    ScopedRootErrorHandler scoped_root_error;
    use_device_ = inp.use_device;

    transporters_.resize(this->num_streams());
    CELER_ENSURE(core_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Run a single step with no active states to "warm up".
 *
 * This is to reduce the uncertainty in timing for problems, especially on AMD
 * hardware.
 */
void Runner::warm_up()
{
    auto& transport = this->get_transporter(StreamId{0});
    transport();
}

//---------------------------------------------------------------------------//
/*!
 * Run on a single stream/thread, returning the transport result.
 *
 * This will partition the input primaries among all the streams.
 */
auto Runner::operator()(StreamId stream, EventId event) -> RunnerResult
{
    CELER_EXPECT(stream < this->num_streams());
    CELER_EXPECT(event < this->num_events());

    auto& transport = this->get_transporter(stream);
    return transport(make_span(events_[event.get()]));
}

//---------------------------------------------------------------------------//
/*!
 * Run all events simultaneously on a single stream.
 */
auto Runner::operator()() -> RunnerResult
{
    CELER_EXPECT(events_.size() == 1);
    CELER_EXPECT(this->num_streams() == 1);

    auto& transport = this->get_transporter(StreamId{0});
    return transport(make_span(events_.front()));
}

//---------------------------------------------------------------------------//
/*!
 * Number of streams supported.
 */
StreamId::size_type Runner::num_streams() const
{
    CELER_EXPECT(core_params_);
    return core_params_->max_streams();
}

//---------------------------------------------------------------------------//
/*!
 * Total number of events.
 */
size_type Runner::num_events() const
{
    return events_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 *
 * This is a *mean* value over all streams.
 *
 * \todo Refactor action times gathering: see celeritas::ActionSequence .
 */
auto Runner::get_action_times() const -> MapStrDouble
{
    MapStrDouble result;
    size_type num_streams{0};
    for (auto sid : range(StreamId{this->num_streams()}))
    {
        if (auto* transport = this->get_transporter_ptr(sid))
        {
            transport->accum_action_times(&result);
            ++num_streams;
        }
    }

    double norm{1 / static_cast<double>(num_streams)};
    for (auto&& [action, time] : result)
    {
        time *= norm;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct transporter input parameters.
 */
void Runner::build_transporter_input(RunnerInput const& inp)
{
    CELER_VALIDATE(inp.max_steps > 0,
                   << "nonpositive max_steps=" << inp.max_steps);

    transporter_input_ = std::make_shared<TransporterInput>();
    transporter_input_->max_steps = inp.max_steps;
    transporter_input_->store_track_counts = inp.write_track_counts;
    transporter_input_->store_step_times = inp.write_step_times;
    transporter_input_->action_times = inp.action_times;
    transporter_input_->params = core_params_;
    transporter_input_->log_progress = inp.log_progress;
}

//---------------------------------------------------------------------------//
/*!
 * Get the transporter for the given stream, constructing if necessary.
 */
auto Runner::get_transporter(StreamId stream) -> TransporterBase&
{
    CELER_EXPECT(stream < transporters_.size());

    UPTransporterBase& result = transporters_[stream.get()];
    if (!result)
    {
        result = [this, stream]() -> std::unique_ptr<TransporterBase> {
            // Thread-local transporter input
            TransporterInput local_trans_inp = *transporter_input_;
            local_trans_inp.stream_id = stream;

            if (use_device_)
            {
                CELER_VALIDATE(device(),
                               << "CUDA device is unavailable but GPU run was "
                                  "requested");
                return std::make_unique<Transporter<MemSpace::device>>(
                    std::move(local_trans_inp));
            }
            else
            {
                return std::make_unique<Transporter<MemSpace::host>>(
                    std::move(local_trans_inp));
            }
        }();
    }
    CELER_ENSURE(result);
    return *result;
}

//---------------------------------------------------------------------------//
/*!
 * Get an already-constructed transporter for the given stream.
 */
auto Runner::get_transporter_ptr(StreamId stream) const -> TransporterBase const*
{
    CELER_EXPECT(stream < transporters_.size());
    return transporters_[stream.get()].get();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

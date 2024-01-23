//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/RootEventReader.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/phys/PrimaryGenerator.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/phys/Process.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/ActionDiagnostic.hh"
#include "celeritas/user/RootStepWriter.hh"
#include "celeritas/user/SimpleCalo.hh"
#include "celeritas/user/StepCollector.hh"
#include "celeritas/user/StepData.hh"
#include "celeritas/user/StepDiagnostic.hh"

#include "RootOutput.hh"
#include "RunnerInput.hh"
#include "Transporter.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get the number of streams from the number of OpenMP threads.
 *
 * The OMP_NUM_THREADS environment variable can be used to control the number
 * of threads/streams. The value of OMP_NUM_THREADS should be a list of
 * positive integers, each of which sets the number of threads for the parallel
 * region at the corresponding nested level. The number of streams is set to
 * the first value in the list. If OMP_NUM_THREADS is not set, the value will
 * be implementation defined.
 */
size_type calc_num_streams(RunnerInput const& inp, size_type num_events)
{
    size_type num_threads = 1;
#if CELERITAS_USE_OPENMP
    if (!inp.merge_events)
    {
#    pragma omp parallel
        {
            if (omp_get_thread_num() == 0)
            {
                num_threads = omp_get_num_threads();
            }
        }
    }
#else
    CELER_DISCARD(inp);
#endif
    // Don't create more streams than events
    return std::min(num_threads, num_events);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
Runner::Runner(RunnerInput const& inp, SPOutputRegistry output)
{
    CELER_EXPECT(output);

    this->setup_globals(inp);

    ScopedRootErrorHandler scoped_root_error;
    this->build_core_params(inp, std::move(output));
    this->build_diagnostics(inp);
    this->build_step_collectors(inp);
    this->build_transporter_input(inp);
    use_device_ = inp.use_device;

    if (root_manager_)
    {
        write_to_root(inp, root_manager_.get());
        write_to_root(*core_params_, root_manager_.get());
    }

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
void Runner::setup_globals(RunnerInput const& inp) const
{
    if (inp.cuda_heap_size != RunnerInput::unspecified)
    {
        set_cuda_heap_size(inp.cuda_heap_size);
    }
    if (inp.cuda_stack_size != RunnerInput::unspecified)
    {
        set_cuda_stack_size(inp.cuda_stack_size);
    }
    environment().merge(inp.environ);
}

//---------------------------------------------------------------------------//
/*!
 * Construct core parameters.
 */
void Runner::build_core_params(RunnerInput const& inp,
                               SPOutputRegistry&& outreg)
{
    CELER_LOG(status) << "Loading input and initializing problem data";
    ScopedMem record_mem("Runner.build_core_params");
    ScopedProfiling profile_this{"construct-params"};
    CoreParams::Input params;
    ImportData const imported = [&inp] {
        if (ends_with(inp.physics_file, ".root"))
        {
            // Load from ROOT file
            return RootImporter(inp.physics_file)();
        }
        std::string filename = inp.physics_file;
        if (filename.empty())
        {
            filename = inp.geometry_file;
        }
        // Load imported data directly from Geant4
        return GeantImporter(GeantSetup(filename, inp.physics_options))();
    }();

    // Create action manager
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::move(outreg);

    // Load geometry
    params.geometry = std::make_shared<GeoParams>(inp.geometry_file);
    if (!params.geometry->supports_safety())
    {
        CELER_LOG(warning) << "Geometry contains surfaces that are "
                              "incompatible with the current ORANGE simple "
                              "safety algorithm: multiple scattering may "
                              "result in arbitrarily small steps";
    }

    // Load materials
    params.material = MaterialParams::from_import(imported);

    // Create geometry/material coupling
    params.geomaterial = GeoMaterialParams::from_import(
        imported, params.geometry, params.material);

    // Construct particle params
    params.particle = ParticleParams::from_import(imported);

    // Construct cutoffs
    params.cutoff = CutoffParams::from_import(
        imported, params.particle, params.material);

    // Load physics: create individual processes with make_shared
    params.physics = [&params, &inp, &imported] {
        PhysicsParams::Input input;
        input.particles = params.particle;
        input.materials = params.material;
        input.action_registry = params.action_reg.get();

        input.options.fixed_step_limiter = inp.step_limiter;
        input.options.secondary_stack_factor = inp.secondary_stack_factor;
        input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
        input.options.lowest_electron_energy = PhysicsParamsOptions::Energy(
            imported.em_params.lowest_electron_energy);

        input.processes = [&params, &inp, &imported] {
            std::vector<std::shared_ptr<Process const>> result;
            ProcessBuilder::Options opts;
            opts.brem_combined = inp.brem_combined;
            opts.brems_selection = inp.physics_options.brems;

            ProcessBuilder build_process(
                imported, params.particle, params.material, opts);
            for (auto p :
                 ProcessBuilder::get_all_process_classes(imported.processes))
            {
                result.push_back(build_process(p));
                CELER_ASSERT(result.back());
            }
            return result;
        }();

        return std::make_shared<PhysicsParams>(std::move(input));
    }();

    bool eloss = imported.em_params.energy_loss_fluct;
    auto msc = UrbanMscParams::from_import(
        *params.particle, *params.material, imported);
    if (inp.field == RunnerInput::no_field())
    {
        // Create along-step action
        auto along_step = AlongStepGeneralLinearAction::from_params(
            params.action_reg->next_id(),
            *params.material,
            *params.particle,
            msc,
            eloss);
        params.action_reg->insert(along_step);
    }
    else
    {
        UniformFieldParams field_params;
        field_params.field = inp.field;
        field_params.options = inp.field_options;

        // Interpret input in units of Tesla
        for (real_type& v : field_params.field)
        {
            v = native_value_from(units::FieldTesla{v});
        }

        auto along_step = AlongStepUniformMscAction::from_params(
            params.action_reg->next_id(),
            *params.material,
            *params.particle,
            field_params,
            msc,
            eloss);
        CELER_ASSERT(along_step->field() != RunnerInput::no_field());
        params.action_reg->insert(along_step);
    }

    // Construct RNG params
    params.rng = std::make_shared<RngParams>(inp.seed);

    // Construct simulation params
    params.sim = SimParams::from_import(imported, params.particle);

    // Get the total number of events
    auto num_events = this->build_events(inp, params.particle);

    // Store the number of simultaneous threads/tasks per process
    params.max_streams = calc_num_streams(inp, num_events);
    CELER_VALIDATE(inp.mctruth_file.empty() || params.max_streams == 1,
                   << "cannot output MC truth with multiple "
                      "streams ("
                   << params.max_streams << " requested)");

    // Construct track initialization params
    params.init = [&inp, &params, num_events] {
        CELER_VALIDATE(inp.initializer_capacity > 0,
                       << "nonpositive initializer_capacity="
                       << inp.initializer_capacity);
        TrackInitParams::Input input;
        input.capacity = ceil_div(inp.initializer_capacity, params.max_streams);
        input.max_events = num_events;
        input.track_order = inp.track_order;
        return std::make_shared<TrackInitParams>(std::move(input));
    }();

    core_params_ = std::make_shared<CoreParams>(std::move(params));
}

//---------------------------------------------------------------------------//
/*!
 * Construct transporter input parameters.
 */
void Runner::build_transporter_input(RunnerInput const& inp)
{
    CELER_VALIDATE(inp.num_track_slots > 0,
                   << "nonpositive num_track_slots=" << inp.num_track_slots);
    CELER_VALIDATE(inp.max_steps > 0,
                   << "nonpositive max_steps=" << inp.max_steps);

    transporter_input_ = std::make_shared<TransporterInput>();
    transporter_input_->num_track_slots
        = ceil_div(inp.num_track_slots, core_params_->max_streams());
    transporter_input_->max_steps = inp.max_steps;
    transporter_input_->store_track_counts = inp.write_track_counts;
    transporter_input_->store_step_times = inp.write_step_times;
    transporter_input_->sync = inp.sync;
    transporter_input_->params = core_params_;
}

//---------------------------------------------------------------------------//
/*!
 * Read events from a file or build using a primary generator.
 *
 * This returns the total number of events.
 */
size_type
Runner::build_events(RunnerInput const& inp, SPConstParticles particles)
{
    ScopedMem record_mem("Runner.build_events");

    if (inp.merge_events)
    {
        // All events will be transported simultaneously on a single stream
        events_.resize(1);
    }

    auto read_events = [&](auto&& generate) {
        auto event = generate();
        while (!event.empty())
        {
            if (inp.merge_events)
            {
                events_.front().insert(
                    events_.front().end(), event.begin(), event.end());
            }
            else
            {
                events_.push_back(event);
            }
            event = generate();
        }
        return generate.num_events();
    };

    if (inp.primary_options)
    {
        return read_events(
            PrimaryGenerator::from_options(particles, inp.primary_options));
    }
    else if (ends_with(inp.event_file, ".root"))
    {
        return read_events(RootEventReader(inp.event_file, particles));
    }
    else
    {
        // Assume filename is one of the HepMC3-supported extensions
        return read_events(EventReader(inp.event_file, particles));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
void Runner::build_step_collectors(RunnerInput const& inp)
{
    StepCollector::VecInterface step_interfaces;
    if (!inp.mctruth_file.empty())
    {
        // Initialize ROOT file
        root_manager_
            = std::make_shared<RootFileManager>(inp.mctruth_file.c_str());

        // Create root step writer
        step_interfaces.push_back(std::make_shared<RootStepWriter>(
            root_manager_,
            core_params_->particle(),
            StepSelection::all(),
            make_write_filter(inp.mctruth_filter)));
    }

    if (!inp.simple_calo.empty())
    {
        auto simple_calo
            = std::make_shared<SimpleCalo>(inp.simple_calo,
                                           *core_params_->geometry(),
                                           core_params_->max_streams());

        // Add to step interfaces
        step_interfaces.push_back(simple_calo);
        // Add to output interface
        core_params_->output_reg()->insert(simple_calo);
    }

    if (!step_interfaces.empty())
    {
        step_collector_ = std::make_unique<StepCollector>(
            std::move(step_interfaces),
            core_params_->geometry(),
            core_params_->max_streams(),
            core_params_->action_reg().get());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct diagnostic actions/outputs.
 */
void Runner::build_diagnostics(RunnerInput const& inp)
{
    if (inp.action_diagnostic)
    {
        auto action_diagnostic = std::make_shared<ActionDiagnostic>(
            core_params_->action_reg()->next_id());

        // Add to action registry
        core_params_->action_reg()->insert(action_diagnostic);
        // Add to output interface
        core_params_->output_reg()->insert(action_diagnostic);
    }

    if (inp.step_diagnostic)
    {
        auto step_diagnostic = std::make_shared<StepDiagnostic>(
            core_params_->action_reg()->next_id(),
            core_params_->particle(),
            inp.step_diagnostic_bins,
            core_params_->max_streams());

        // Add to action registry
        core_params_->action_reg()->insert(step_diagnostic);
        // Add to output interface
        core_params_->output_reg()->insert(step_diagnostic);
    }
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
auto Runner::get_transporter_ptr(StreamId stream) const
    -> TransporterBase const*
{
    CELER_EXPECT(stream < transporters_.size());
    return transporters_[stream.get()].get();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Runner.cc
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include <functional>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedMem.hh"
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

#include "RootOutput.hh"
#include "RunnerInput.hh"
#include "Transporter.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
Runner::Runner(RunnerInput const& inp,
               std::shared_ptr<celeritas::OutputRegistry> output)
{
    CELER_EXPECT(output);

    this->setup_globals(inp);

    ScopedRootErrorHandler scoped_root_error;
    this->build_core_params(inp, std::move(output));
    this->build_step_collectors(inp);
    this->build_diagnostics(inp);
    this->build_transporter_input(inp);
    this->build_primaries(inp);
    use_device_ = inp.use_device;

    CELER_ENSURE(core_params_);
}

//---------------------------------------------------------------------------//
/*!
 * Run on a single stream/thread, returning the transport result.
 *
 * This will partition the input primaries among all the streams.
 */
auto Runner::operator()(StreamId stream_id) const -> RunnerResult
{
    CELER_EXPECT(stream_id < this->num_streams());

    auto transport = [this, stream_id]() -> std::unique_ptr<TransporterBase> {
        // Thread-local transporter input
        TransporterInput local_trans_inp = *transporter_input_;
        local_trans_inp.stream_id = stream_id;

        if (use_device_)
        {
            CELER_VALIDATE(celeritas::device(),
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

    // TODO: partition primaries among streams
    CELER_ASSERT(stream_id == StreamId{0});
    return (*transport)(make_span(primaries_));
}

//---------------------------------------------------------------------------//
/*!
 * Number of streams supported.
 */
StreamId::size_type Runner::num_streams() const
{
    return core_params_->max_streams();
}

//---------------------------------------------------------------------------//
void Runner::setup_globals(RunnerInput const& inp) const
{
    if (inp.cuda_heap_size != RunnerInput::unspecified)
    {
        celeritas::set_cuda_heap_size(inp.cuda_heap_size);
    }
    if (inp.cuda_stack_size != RunnerInput::unspecified)
    {
        celeritas::set_cuda_stack_size(inp.cuda_stack_size);
    }
    celeritas::environment().merge(inp.environ);
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
    CoreParams::Input params;
    ImportData const imported = [&inp] {
        if (ends_with(inp.physics_filename, ".root"))
        {
            // Load imported from ROOT file
            return RootImporter(inp.physics_filename.c_str())();
        }
        else if (ends_with(inp.physics_filename, ".gdml"))
        {
            // Load imported directly from Geant4
            return GeantImporter(
                GeantSetup(inp.physics_filename, inp.geant_options))();
        }
        CELER_VALIDATE(false,
                       << "invalid physics filename '" << inp.physics_filename
                       << "' (expected gdml or root)");
    }();

    // Create action manager
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::move(outreg);

    // Load geometry
    params.geometry
        = std::make_shared<GeoParams>(inp.geometry_filename.c_str());
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
        input.options.lowest_electron_energy = PhysicsParamsOptions::Energy{
            imported.em_params.lowest_electron_energy};

        input.processes = [&params, &inp, &imported] {
            std::vector<std::shared_ptr<Process const>> result;
            ProcessBuilder::Options opts;
            opts.brem_combined = inp.brem_combined;

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
    if (inp.mag_field == RunnerInput::no_field())
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
        CELER_VALIDATE(!eloss,
                       << "energy loss fluctuations are not supported "
                          "simultaneoulsy with magnetic field");
        UniformFieldParams field_params;
        field_params.field = inp.mag_field;
        field_params.options = inp.field_options;

        // Interpret input in units of Tesla
        for (real_type& f : field_params.field)
        {
            f *= units::tesla;
        }

        auto along_step = std::make_shared<AlongStepUniformMscAction>(
            params.action_reg->next_id(), field_params, msc);
        CELER_ASSERT(along_step->field() != RunnerInput::no_field());
        params.action_reg->insert(along_step);
    }

    // Construct RNG params
    params.rng = std::make_shared<RngParams>(inp.seed);

    // Construct simulation params
    params.sim = SimParams::from_import(imported, params.particle);

    // Construct track initialization params
    params.init = [&inp] {
        CELER_VALIDATE(inp.initializer_capacity > 0,
                       << "nonpositive initializer_capacity="
                       << inp.initializer_capacity);
        CELER_VALIDATE(inp.max_events > 0,
                       << "nonpositive max_events=" << inp.max_events);
        CELER_VALIDATE(
            !inp.primary_gen_options
                || inp.max_events >= inp.primary_gen_options.num_events,
            << "max_events=" << inp.max_events
            << " cannot be less than num_events="
            << inp.primary_gen_options.num_events);
        TrackInitParams::Input input;
        input.capacity = inp.initializer_capacity;
        input.max_events = inp.max_events;
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
    CELER_VALIDATE(inp.max_num_tracks > 0,
                   << "nonpositive max_num_tracks=" << inp.max_num_tracks);
    CELER_VALIDATE(inp.max_steps > 0,
                   << "nonpositive max_steps=" << inp.max_steps);

    transporter_input_ = std::make_shared<TransporterInput>();
    transporter_input_->num_track_slots = inp.max_num_tracks;
    transporter_input_->max_steps = inp.max_steps;
    transporter_input_->enable_diagnostics = inp.enable_diagnostics;
    transporter_input_->sync = inp.sync;
    transporter_input_->params = core_params_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
void Runner::build_primaries(RunnerInput const& inp)
{
    ScopedMem record_mem("Runner.build_primaries");
    if (inp.primary_gen_options)
    {
        std::mt19937 rng;
        auto generate_event = PrimaryGenerator<std::mt19937>::from_options(
            core_params_->particle(), inp.primary_gen_options);
        auto event = generate_event(rng);
        while (!event.empty())
        {
            primaries_.insert(primaries_.end(), event.begin(), event.end());
            event = generate_event(rng);
        }
    }
    else
    {
        EventReader read_event(inp.hepmc3_filename.c_str(),
                               core_params_->particle());
        auto event = read_event();
        while (!event.empty())
        {
            primaries_.insert(primaries_.end(), event.begin(), event.end());
            event = read_event();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct on all threads from a JSON input and shared output manager.
 */
void Runner::build_step_collectors(RunnerInput const& inp)
{
    StepCollector::VecInterface step_interfaces;
    if (!inp.mctruth_filename.empty())
    {
        // Initialize ROOT file; Store input and core params
        root_manager_
            = std::make_shared<RootFileManager>(inp.mctruth_filename.c_str());
        write_to_root(inp, root_manager_.get());
        write_to_root(*core_params_, root_manager_.get());

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
            core_params_->action_reg()->next_id(),
            core_params_->action_reg(),
            core_params_->particle(),
            core_params_->max_streams());

        // Add to action registry
        core_params_->action_reg()->insert(action_diagnostic);
        // Add to output interface
        core_params_->output_reg()->insert(action_diagnostic);
    }
}

//---------------------------------------------------------------------------//
}  // namespace demo_loop

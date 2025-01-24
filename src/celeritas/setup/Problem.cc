//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Problem.cc
//---------------------------------------------------------------------------//

namespace celeritas
{
namespace setup
{
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
                               G4VPhysicalVolume const* g4world,
                               ImportData const& imported)
{
    CELER_LOG(status) << "Loading input and initializing problem data";
    ScopedMem record_mem("Runner.build_core_params");
    ScopedProfiling profile_this{"construct-params"};
    CoreParams::Input params;

    // Create action manager
    params.action_reg = std::make_shared<ActionRegistry>();
    params.output_reg = std::make_shared<OutputRegistry>();

    // Load geometry: use existing world volume or reload from geometry file
    params.geometry = [&geo_file = inp.geometry_file, g4world] {
        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            static char const fi_hack_envname[] = "ORANGE_FORCE_INPUT";
            auto const& filename = celeritas::getenv(fi_hack_envname);
            if (!filename.empty())
            {
                CELER_LOG(warning)
                    << "Using a temporary, unsupported, and dangerous hack to "
                       "override the ORANGE geometry file: "
                    << fi_hack_envname << "='" << filename << "'";
                return std::make_shared<GeoParams>(filename);
            }
        }
        if (g4world)
        {
            return std::make_shared<GeoParams>(g4world);
        }
        return std::make_shared<GeoParams>(geo_file);
    }();

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

    // Construct shared data for Coulomb scattering
    params.wentzel = WentzelOKVIParams::from_import(
        imported, params.material, params.particle);

    // Load physics: create individual processes with make_shared
    params.physics = [&params, &inp, &imported] {
        PhysicsParams::Input input;
        input.particles = params.particle;
        input.materials = params.material;
        input.action_registry = params.action_reg.get();

        // Set physics options
        input.options.fixed_step_limiter = inp.step_limiter;
        input.options.secondary_stack_factor = inp.secondary_stack_factor;
        input.options.spline_eloss_order = inp.spline_eloss_order;
        input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
        input.options.light.lowest_energy = ParticleOptions::Energy(
            imported.em_params.lowest_electron_energy);
        input.options.heavy.lowest_energy
            = ParticleOptions::Energy(imported.em_params.lowest_muhad_energy);

        // Set multiple scattering options
        input.options.light.range_factor = imported.em_params.msc_range_factor;
        input.options.heavy.range_factor
            = imported.em_params.msc_muhad_range_factor;
        input.options.safety_factor = imported.em_params.msc_safety_factor;
        input.options.lambda_limit = imported.em_params.msc_lambda_limit;
        input.options.light.displaced = imported.em_params.msc_displaced;
        input.options.heavy.displaced = imported.em_params.msc_muhad_displaced;
        input.options.light.step_limit_algorithm
            = imported.em_params.msc_step_algorithm;
        input.options.heavy.step_limit_algorithm
            = imported.em_params.msc_muhad_step_algorithm;

        // Build processes
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
    params.sim = std::make_shared<SimParams>([&] {
        // TODO: use max_steps here instead of as step iteration?
        auto input = SimParams::Input::from_import(
            imported, params.particle, inp.field_options.max_substeps);
        return input;
    }());

    // Get the total number of events
    auto num_events = this->build_events(inp, params.particle);

    // Store the number of simultaneous threads/tasks per process
    params.max_streams = calc_num_streams(inp, num_events);
    CELER_VALIDATE(inp.mctruth_file.empty() || params.max_streams == 1,
                   << "cannot output MC truth with multiple "
                      "streams ("
                   << params.max_streams << " requested)");

    // Store number of tracks per stream
    CELER_VALIDATE(inp.num_track_slots > 0,
                   << "nonpositive num_track_slots=" << inp.num_track_slots);
    params.tracks_per_stream
        = ceil_div(inp.num_track_slots, params.max_streams);

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
            core_params_->geometry(),
            std::move(step_interfaces),
            core_params_->aux_reg().get(),
            core_params_->action_reg().get());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct optical collector.
 *
 * \pre Must be called after \c build_core_params .
 */
void Runner::build_optical_collector(RunnerInput const& inp,
                                     ImportData const& imported)
{
    CELER_EXPECT(core_params_);

    using optical::CherenkovParams;
    using optical::MaterialParams;
    using optical::ScintillationParams;

    //! \todo Update conditionals after implementing CelerOpticalPhysicsList
    if (imported.optical_materials.empty())
    {
        // No optical materials are present
        return;
    }
    CELER_ASSERT(inp.optical);

    size_type num_streams = core_params_->max_streams();

    OpticalCollector::Input oc_inp;
    oc_inp.material = MaterialParams::from_import(
        imported, *core_params_->geomaterial(), *core_params_->material());
    oc_inp.cherenkov = std::make_shared<CherenkovParams>(*oc_inp.material);
    oc_inp.scintillation
        = ScintillationParams::from_import(imported, core_params_->particle());
    oc_inp.num_track_slots = ceil_div(inp.optical.num_track_slots, num_streams);
    oc_inp.buffer_capacity = ceil_div(inp.optical.buffer_capacity, num_streams);
    oc_inp.initializer_capacity
        = ceil_div(inp.optical.initializer_capacity, num_streams);
    oc_inp.auto_flush = ceil_div(inp.optical.auto_flush, num_streams);

    CELER_ASSERT(oc_inp);
    optical_collector_
        = std::make_shared<OpticalCollector>(*core_params_, std::move(oc_inp));
}

//---------------------------------------------------------------------------//
/*!
 * Construct diagnostic actions/outputs.
 */
void Runner::build_diagnostics(RunnerInput const& inp)
{
    if (inp.action_diagnostic)
    {
        ActionDiagnostic::make_and_insert(*core_params_);
    }

    if (inp.step_diagnostic)
    {
        StepDiagnostic::make_and_insert(*core_params_,
                                        inp.step_diagnostic_bins);
    }

    if (!inp.slot_diagnostic_prefix.empty())
    {
        SlotDiagnostic::make_and_insert(*core_params_,
                                        inp.slot_diagnostic_prefix);
    }
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas

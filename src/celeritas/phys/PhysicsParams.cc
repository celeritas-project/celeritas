//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include <algorithm>
#include <map>
#include <tuple>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/VectorUtils.hh"
#include "celeritas/em/AtomicRelaxationParams.hh"
#include "celeritas/em/model/CombinedBremModel.hh"
#include "celeritas/em/model/EPlusGGModel.hh"
#include "celeritas/em/model/LivermorePEModel.hh"
#include "celeritas/em/model/UrbanMscModel.hh"
#include "celeritas/em/process/MultipleScatteringProcess.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/global/generated/AlongStepAction.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridInserter.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "ParticleParams.hh"
#include "generated/DiscreteSelectAction.hh"
#include "generated/PreStepAction.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
class ImplicitPhysicsAction final : public ImplicitActionInterface,
                                    public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;
};
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with processes and helper classes.
 */
PhysicsParams::PhysicsParams(Input inp)
    : processes_(std::move(inp.processes))
    , relaxation_(std::move(inp.relaxation))
{
    CELER_EXPECT(!processes_.empty());
    CELER_EXPECT(std::all_of(processes_.begin(),
                             processes_.end(),
                             [](const SPConstProcess& p) { return bool(p); }));
    CELER_EXPECT(inp.particles);
    CELER_EXPECT(inp.materials);
    CELER_EXPECT(inp.action_manager);

    // Create actions (order matters due to accessors in PhysicsParamsScalars)
    {
        using std::make_shared;
        auto& action_mgr = *inp.action_manager;
        // TODO: add scoped range
        // auto  get_action_range = action_mgr.scoped_range("physics");

        auto pre_step_action = make_shared<generated::PreStepAction>(
            action_mgr.next_id(), "pre-step", "beginning of step physics");
        inp.action_manager->insert(pre_step_action);
        pre_step_action_ = std::move(pre_step_action);

        // TODO: this is coupled to geometry and field, so find a better place
        // for this
        auto along_step_action = make_shared<generated::AlongStepAction>(
            action_mgr.next_id(),
            "along-step",
            "propagation, multiple scattering, and energy loss");
        inp.action_manager->insert(along_step_action);
        along_step_action_ = std::move(along_step_action);

        auto range_action = make_shared<ImplicitPhysicsAction>(
            action_mgr.next_id(),
            "eloss-range",
            "range limitation due to energy loss");
        action_mgr.insert(range_action);
        range_action_ = std::move(range_action);

        auto discrete_action = make_shared<generated::DiscreteSelectAction>(
            action_mgr.next_id(),
            "physics-discrete-select",
            "discrete interaction");
        inp.action_manager->insert(discrete_action);
        discrete_action_ = std::move(discrete_action);

        auto integral_action = make_shared<ImplicitPhysicsAction>(
            action_mgr.next_id(),
            "physics-integral-rejected",
            "rejection by integral cross section");
        inp.action_manager->insert(integral_action);
        integral_rejection_action_ = std::move(integral_action);

        // Emit models for associated proceses
        models_ = this->build_models(inp.action_manager);

        // Place "failure" *after* all the model IDs
        auto failure_action = make_shared<ImplicitPhysicsAction>(
            action_mgr.next_id(),
            "physics-failure",
            "interaction sampling failure");
        inp.action_manager->insert(failure_action);
        failure_action_ = std::move(failure_action);
    }

    // Construct data
    HostValue host_data;
    this->build_options(inp.options, &host_data);
    this->build_fluct(inp.options, *inp.materials, *inp.particles, &host_data);
    this->build_ids(*inp.particles, &host_data);
    this->build_xs(inp.options, *inp.materials, &host_data);
    this->build_model_xs(*inp.materials, &host_data);

    // Add step limiter if being used (TODO: remove this hack from physics)
    if (inp.options.fixed_step_limiter > 0)
    {
        using std::make_shared;
        auto& action_mgr = *inp.action_manager;

        auto fixed_step_action = make_shared<ImplicitPhysicsAction>(
            action_mgr.next_id(),
            "physics-fixed-step",
            "fixed step limiter for charged particles");
        inp.action_manager->insert(fixed_step_action);
        host_data.scalars.fixed_step_limiter = inp.options.fixed_step_limiter;
        host_data.scalars.fixed_step_action  = fixed_step_action->action_id();
        fixed_step_action_                   = std::move(fixed_step_action);
    }

    // Copy data to device
    data_ = CollectionMirror<PhysicsParamsData>{std::move(host_data)};

    CELER_ENSURE(range_action_->action_id()
                 == host_ref().scalars.range_action());
    CELER_ENSURE(discrete_action_->action_id()
                 == host_ref().scalars.discrete_action());
    CELER_ENSURE(integral_rejection_action_->action_id()
                 == host_ref().scalars.integral_rejection_action());
    CELER_ENSURE(failure_action_->action_id()
                 == host_ref().scalars.failure_action());
}

//---------------------------------------------------------------------------//
/*!
 * Get the list of process IDs that apply to a particle type.
 */
auto PhysicsParams::processes(ParticleId id) const -> SpanConstProcessId
{
    CELER_EXPECT(id < this->num_processes());
    const auto& data = this->host_ref();
    return data.process_ids[data.process_groups[id].processes];
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
auto PhysicsParams::build_models(ActionManager* mgr) const -> VecModel
{
    VecModel models;

    // Construct models, assigning each model ID
    for (auto process_idx : range<ProcessId::size_type>(processes_.size()))
    {
        auto id_iter    = Process::ActionIdIter{mgr->next_id()};
        auto new_models = processes_[process_idx]->build_models(id_iter);
        CELER_ASSERT(!new_models.empty());
        for (SPConstModel& model : new_models)
        {
            CELER_ASSERT(model);
            CELER_ASSERT(model->action_id() == *id_iter++);

            // Add model to action manager
            mgr->insert(model);
            // Save model and the process that it belongs to
            models.push_back({std::move(model), ProcessId{process_idx}});
        }
    }

    CELER_ENSURE(!models.empty());
    return models;
}

//---------------------------------------------------------------------------//
/*!
 * Construct on-device physics options.
 */
void PhysicsParams::build_options(const Options& opts, HostValue* data) const
{
    CELER_VALIDATE(opts.max_step_over_range > 0,
                   << "invalid max_step_over_range="
                   << opts.max_step_over_range << " (should be positive)");
    CELER_VALIDATE(opts.min_eprime_over_e > 0 && opts.min_eprime_over_e < 1,
                   << "invalid min_eprime_over_e=" << opts.min_eprime_over_e
                   << " (should be be within 0 < limit < 1)");
    CELER_VALIDATE(opts.min_range > 0,
                   << "invalid min_range=" << opts.min_range
                   << " (should be positive)");
    CELER_VALIDATE(opts.linear_loss_limit >= 0 && opts.linear_loss_limit <= 1,
                   << "invalid linear_loss_limit=" << opts.linear_loss_limit
                   << " (should be within 0 <= limit <= 1)");
    CELER_VALIDATE(opts.secondary_stack_factor > 0,
                   << "invalid secondary_stack_factor=" << opts.secondary_stack_factor
                   << " (should be positive)");
    data->scalars.scaling_min_range  = opts.min_range;
    data->scalars.scaling_fraction   = opts.max_step_over_range;
    data->scalars.energy_fraction    = opts.min_eprime_over_e;
    data->scalars.linear_loss_limit  = opts.linear_loss_limit;
    data->scalars.enable_fluctuation = opts.enable_fluctuation;
    data->scalars.secondary_stack_factor = opts.secondary_stack_factor;
}

//---------------------------------------------------------------------------//
/*!
 * Construct particle -> process -> model mappings.
 */
void PhysicsParams::build_ids(const ParticleParams& particles,
                              HostValue*            data) const
{
    CELER_EXPECT(data);
    CELER_EXPECT(!models_.empty());
    using ModelRange = std::tuple<real_type, real_type, ParticleModelId>;

    // Offset from the index in the list of models to a model's ActionId
    data->scalars.model_to_action = this->model(ModelId{0}).action_id().get();

    // Note: use map to keep ProcessId sorted
    std::vector<std::map<ProcessId, std::vector<ModelRange>>> particle_models(
        particles.size());
    std::vector<ModelId>       temp_model_ids;
    ParticleModelId::size_type pm_idx{0};

    // Construct particle -> process -> model map
    for (auto model_idx : range(this->num_models()))
    {
        const Model&    m          = *models_[model_idx].first;
        const ProcessId process_id = models_[model_idx].second;
        for (const Applicability& applic : m.applicability())
        {
            if (applic.material)
            {
                CELER_NOT_IMPLEMENTED("Material-dependent models");
            }
            CELER_VALIDATE(applic.particle < particles.size(),
                           << "invalid particle ID "
                           << applic.particle.unchecked_get());
            CELER_ASSERT(applic.lower < applic.upper);
            particle_models[applic.particle.get()][process_id].push_back(
                {value_as<ModelGroup::Energy>(applic.lower),
                 value_as<ModelGroup::Energy>(applic.upper),
                 ParticleModelId{pm_idx++}});
            temp_model_ids.push_back(ModelId{model_idx});
        }
    }
    make_builder(&data->model_ids)
        .insert_back(temp_model_ids.begin(), temp_model_ids.end());

    auto process_groups = make_builder(&data->process_groups);
    auto process_ids    = make_builder(&data->process_ids);
    auto model_groups   = make_builder(&data->model_groups);
    auto pmodel_ids     = make_builder(&data->pmodel_ids);
    auto reals          = make_builder(&data->reals);

    process_groups.reserve(particle_models.size());

    // Loop over particle IDs, set ProcessGroup
    ProcessId::size_type max_particle_processes = 0;
    for (auto particle_idx : range(particles.size()))
    {
        auto& process_to_models = particle_models[particle_idx];
        if (process_to_models.empty())
        {
            CELER_LOG(warning)
                << "No processes are defined for particle '"
                << particles.id_to_label(ParticleId{particle_idx}) << '\'';
        }
        max_particle_processes = std::max<ProcessId::size_type>(
            max_particle_processes, process_to_models.size());

        std::vector<ProcessId>  temp_processes;
        std::vector<ModelGroup> temp_model_datas;
        temp_processes.reserve(process_to_models.size());
        temp_model_datas.reserve(process_to_models.size());
        for (auto& pid_models : process_to_models)
        {
            // Add process ID
            temp_processes.push_back(pid_models.first);

            std::vector<ModelRange>& models = pid_models.second;
            CELER_ASSERT(!models.empty());

            // Construct model data
            std::vector<real_type>       temp_energy_grid;
            std::vector<ParticleModelId> temp_models;
            temp_energy_grid.reserve(models.size() + 1);
            temp_models.reserve(models.size());

            // Sort, and add the first grid point
            std::sort(models.begin(), models.end());
            temp_energy_grid.push_back(std::get<0>(models[0]));

            for (const ModelRange& r : models)
            {
                CELER_VALIDATE(
                    temp_energy_grid.back() == std::get<0>(r),
                    << "models for process '"
                    << this->process(pid_models.first).label()
                    << "' of particle type '"
                    << particles.id_to_label(ParticleId{particle_idx})
                    << "' has no data between energies of "
                    << temp_energy_grid.back() << " and " << std::get<0>(r)
                    << " (energy range must be contiguous)");
                temp_energy_grid.push_back(std::get<1>(r));
                temp_models.push_back(std::get<2>(r));
            }

            ModelGroup mdata;
            mdata.energy = reals.insert_back(temp_energy_grid.begin(),
                                             temp_energy_grid.end());
            mdata.model  = pmodel_ids.insert_back(temp_models.begin(),
                                                 temp_models.end());
            CELER_ASSERT(mdata);
            temp_model_datas.push_back(mdata);
        }

        ProcessGroup pdata;
        pdata.processes = process_ids.insert_back(temp_processes.begin(),
                                                  temp_processes.end());
        pdata.models    = model_groups.insert_back(temp_model_datas.begin(),
                                                temp_model_datas.end());

        // It's ok to have particles defined in the problem that do not have
        // any processes (if they are ever created, they will just be
        // transported until they exit the geometry).
        // NOTE: data tables will be assigned later
        CELER_ASSERT(process_to_models.empty() || pdata);
        process_groups.push_back(pdata);
    }
    data->scalars.max_particle_processes = max_particle_processes;
    data->scalars.num_models             = this->num_models();

    // Assign hardwired models that do on-the-fly xs calculation
    for (auto model_idx : range(this->num_models()))
    {
        const Model&    model      = *models_[model_idx].first;
        const ProcessId process_id = models_[model_idx].second;
        if (auto* pe_model = dynamic_cast<const LivermorePEModel*>(&model))
        {
            data->hardwired.photoelectric              = process_id;
            data->hardwired.photoelectric_table_thresh = units::MevEnergy{0.2};
            data->hardwired.livermore_pe               = ModelId{model_idx};
            data->hardwired.livermore_pe_data          = pe_model->host_ref();
        }
        else if (auto* epgg_model = dynamic_cast<const EPlusGGModel*>(&model))
        {
            data->hardwired.positron_annihilation = process_id;
            data->hardwired.eplusgg               = ModelId{model_idx};
            data->hardwired.eplusgg_data          = epgg_model->device_ref();
        }
        else if (auto* urban_model = dynamic_cast<const UrbanMscModel*>(&model))
        {
            data->hardwired.msc        = process_id;
            data->hardwired.urban      = ModelId{model_idx};
            data->hardwired.urban_data = urban_model->host_ref();
        }
    }

    if (relaxation_)
    {
        data->hardwired.relaxation_data = relaxation_->host_ref();
    }

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
/*!
 * Construct cross section data.
 */
void PhysicsParams::build_xs(const Options&        opts,
                             const MaterialParams& mats,
                             HostValue*            data) const
{
    CELER_EXPECT(*data);

    using UPGridBuilder = Process::UPConstGridBuilder;
    using Energy        = Applicability::Energy;

    ValueGridInserter insert_grid(&data->reals, &data->value_grids);
    auto              value_tables   = make_builder(&data->value_tables);
    auto              integral_xs    = make_builder(&data->integral_xs);
    auto              value_grid_ids = make_builder(&data->value_grid_ids);
    auto              build_grid
        = [insert_grid](const UPGridBuilder& builder) -> ValueGridId {
        return builder ? builder->build(insert_grid) : ValueGridId{};
    };

    Applicability applic;
    for (auto particle_id : range(ParticleId(data->process_groups.size())))
    {
        applic.particle = particle_id;

        // Processes for this particle
        ProcessGroup& process_groups = data->process_groups[particle_id];
        Span<const ProcessId> processes
            = data->process_ids[process_groups.processes];
        Span<const ModelGroup> model_groups
            = data->model_groups[process_groups.models];
        CELER_ASSERT(processes.size() == model_groups.size());

        // Material-dependent physics tables, one per particle-process
        ValueGridArray<std::vector<ValueTable>> temp_tables;
        for (auto& vec : temp_tables)
        {
            vec.resize(processes.size());
        }

        // Processes with dE/dx and macro xs tables
        std::vector<IntegralXsProcess> temp_integral_xs(processes.size());

        // Loop over per-particle processes
        for (auto pp_idx :
             range(ParticleProcessId::size_type(processes.size())))
        {
            // Get energy bounds for this process
            Span<const real_type> energy_grid
                = data->reals[model_groups[pp_idx].energy];
            applic.lower = Energy{energy_grid.front()};
            applic.upper = Energy{energy_grid.back()};
            CELER_ASSERT(applic.lower < applic.upper);

            const Process& proc = this->process(processes[pp_idx]);

            // Grid IDs for each grid type, each material
            ValueGridArray<std::vector<ValueGridId>> temp_grid_ids;
            for (auto& vec : temp_grid_ids)
            {
                vec.resize(mats.size());
            }

            // Energy of maximum cross section for each material
            std::vector<real_type> energy_max_xs;
            bool                   use_integral_xs = !opts.disable_integral_xs
                                   && proc.use_integral_xs();
            if (use_integral_xs)
            {
                energy_max_xs.resize(mats.size());
            }

            // Loop over materials
            for (auto mat_id : range(MaterialId{mats.size()}))
            {
                applic.material = mat_id;

                // Construct step limit builders
                auto builders = proc.step_limits(applic);
                CELER_VALIDATE(
                    std::any_of(builders.begin(),
                                builders.end(),
                                [](const UPGridBuilder& p) { return bool(p); }),
                    << "process '" << proc.label()
                    << "' has neither interaction nor energy loss (it must "
                       "have at least one)");

                // Construct grids
                for (auto vgt : range(ValueGridType::size_))
                {
                    temp_grid_ids[vgt][mat_id.get()]
                        = build_grid(builders[vgt]);
                }

                if (processes[pp_idx] == data->hardwired.positron_annihilation)
                {
                    // Discrete interaction can occur at rest
                    process_groups.has_at_rest = true;

                    if (use_integral_xs)
                    {
                        // Annihilation cross section is maximum at zero and
                        // decreases with increasing energy
                        energy_max_xs[mat_id.get()] = 0;
                    }
                }
                else if (auto grid_id
                         = temp_grid_ids[ValueGridType::macro_xs][mat_id.get()])
                {
                    const auto&        grid_data = data->value_grids[grid_id];
                    auto               data_ref  = make_const_ref(*data);
                    const UniformGrid  loge_grid(grid_data.log_energy);
                    const XsCalculator calc_xs(grid_data, data_ref.reals);

                    // Check if the particle can have a discrete interaction at
                    // rest
                    process_groups.has_at_rest |= calc_xs(zero_quantity()) > 0;

                    // Find and store the energy of the largest cross section
                    // for this material if the integral approach is used
                    if (use_integral_xs)
                    {
                        // Find the energy of the largest cross section
                        real_type xs_max = 0;
                        real_type e_max  = 0;
                        for (auto i : range(loge_grid.size()))
                        {
                            real_type xs = calc_xs[i];
                            if (xs > xs_max)
                            {
                                xs_max = xs;
                                e_max  = std::exp(loge_grid[i]);
                            }
                        }
                        CELER_ASSERT(e_max > 0);
                        energy_max_xs[mat_id.get()] = e_max;
                    }
                }

                // Index of the energy loss process that stores the de/dx and
                // range tables
                if (temp_grid_ids[ValueGridType::energy_loss][mat_id.get()]
                    && temp_grid_ids[ValueGridType::range][mat_id.get()])
                {
                    // Only one particle-process should have energy loss tables
                    CELER_ASSERT(!process_groups.eloss_ppid
                                 || pp_idx == process_groups.eloss_ppid.get());
                    process_groups.eloss_ppid = ParticleProcessId{pp_idx};
                }

                // Index of the electromagnetic msc process
                if (dynamic_cast<const MultipleScatteringProcess*>(&proc))
                {
                    process_groups.msc_ppid = ParticleProcessId{pp_idx};
                }
            }

            // Outer loop over grid types
            for (auto vgt : range(ValueGridType::size_))
            {
                if (!std::any_of(temp_grid_ids[vgt].begin(),
                                 temp_grid_ids[vgt].end(),
                                 [](ValueGridId id) { return bool(id); }))
                {
                    if (vgt == ValueGridType::macro_xs)
                    {
                        // Skip this table type since it's not present for any
                        // material for this particle process
                        CELER_LOG(debug)
                            << "No " << to_cstring(ValueGridType(vgt))
                            << " for process " << proc.label();
                    }
                    continue;
                }

                // Construct value grid table
                ValueTable& temp_table = temp_tables[vgt][pp_idx];
                temp_table.grids       = value_grid_ids.insert_back(
                    temp_grid_ids[vgt].begin(), temp_grid_ids[vgt].end());
                CELER_ASSERT(temp_table.grids.size() == mats.size());
            }

            // Store the energies of the maximum cross sections
            if (!energy_max_xs.empty())
            {
                temp_integral_xs[pp_idx].energy_max_xs
                    = make_builder(&data->reals)
                          .insert_back(energy_max_xs.begin(),
                                       energy_max_xs.end());
            }
        }

        // Construct energy loss process data
        process_groups.integral_xs = integral_xs.insert_back(
            temp_integral_xs.begin(), temp_integral_xs.end());

        // Construct value tables
        for (auto vgt : range(ValueGridType::size_))
        {
            process_groups.tables[vgt] = value_tables.insert_back(
                temp_tables[vgt].begin(), temp_tables[vgt].end());
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct model cross section CDFs.
 */
void PhysicsParams::build_model_xs(const MaterialParams& mats,
                                   HostValue*            data) const
{
    CELER_EXPECT(*data);

    ValueGridInserter insert_grid(&data->reals, &data->value_grids);

    // Micro xs grid IDs for each model and applicable particle, each material,
    // and each element in the material
    std::vector<std::vector<std::vector<ValueGridId>>> temp_grid_ids(
        data->model_ids.size());
    size_type pm_idx{0};

    for (auto model_idx : range(this->num_models()))
    {
        const Model& model = *models_[model_idx].first;

        // Loop over applicable particles
        for (Applicability applic : model.applicability())
        {
            for (auto mat_id : range(MaterialId{mats.size()}))
            {
                applic.material = mat_id;
                auto material   = mats.get(mat_id);

                // TODO: Create combined SB + RB micro xs grids or possibly
                // remove combined bremsstrahlung model
                CELER_VALIDATE(!(dynamic_cast<const CombinedBremModel*>(&model)
                                 && material.num_elements() > 1),
                               << "model '" << model.label()
                               << "' cannot be used with materials composed "
                                  "of more than one element (material '"
                               << mats.id_to_label(mat_id) << "' has "
                               << material.num_elements() << " elements)");

                // Construct microscopic cross section builders
                auto builders = model.micro_xs(applic);
                if (builders.empty())
                {
                    // Models that calculate xs on the fly and models
                    // with material-independent discrete interactions
                    // won't have micro xs grids
                    continue;
                }
                CELER_ASSERT(builders.size() == material.num_elements());

                // Construct grids for each element in the material
                CELER_ASSERT(pm_idx < temp_grid_ids.size());
                temp_grid_ids[pm_idx].resize(mats.size());
                if (material.num_elements() > 1)
                {
                    auto& grid_ids = temp_grid_ids[pm_idx][mat_id.get()];
                    grid_ids.resize(material.num_elements());

                    for (auto elcomp_id :
                         range(ElementComponentId{material.num_elements()}))
                    {
                        auto el_id = material.element_id(elcomp_id);
                        auto iter  = builders.find(el_id);
                        CELER_ASSERT(iter != builders.end());
                        CELER_ASSERT(iter->second);
                        grid_ids[elcomp_id.get()]
                            = iter->second->build(insert_grid);
                    }
                }
            }
            ++pm_idx;
        }
    }

    auto model_xs        = make_builder(&data->model_xs);
    auto value_tables    = make_builder(&data->value_tables);
    auto value_table_ids = make_builder(&data->value_table_ids);
    auto value_grid_ids  = make_builder(&data->value_grid_ids);

    // Construct model cross section CDF tables
    for (auto& model_table : temp_grid_ids)
    {
        std::vector<ValueTableId> temp_table_ids(model_table.size());
        for (auto mat_idx : range<MaterialId::size_type>(model_table.size()))
        {
            auto& grid_ids = model_table[mat_idx];
            if (grid_ids.empty())
            {
                // No micro xs stored for this material
                continue;
            }

            // Get the xs value for the given element and bin
            auto get_value = [&](size_type elcomp, size_type bin) -> real_type& {
                XsGridData& grid = data->value_grids[grid_ids[elcomp]];
                CELER_ASSERT(bin < grid.value.size());
                return data->reals[grid.value[bin]];
            };

            // Get the number of grid points: the energy grids are the
            // same for each element in the material
            size_type num_bins = data->value_grids[grid_ids[0]].value.size();

            // Calculate the cross section CDF
            const auto elements = mats.get(MaterialId{mat_idx}).elements();
            for (auto bin_idx : range(num_bins))
            {
                real_type cum_xs{0};
                for (auto elcomp_idx : range(elements.size()))
                {
                    real_type& xs = get_value(elcomp_idx, bin_idx);
                    cum_xs += xs * elements[elcomp_idx].fraction;
                    xs = cum_xs;
                }

                // Normalize
                if (cum_xs > 0)
                {
                    for (auto elcomp_idx : range(elements.size()))
                    {
                        real_type& xs = get_value(elcomp_idx, bin_idx);
                        xs /= cum_xs;
                    }
                }
            }
            // Construct value grid table
            ValueTable temp_table;
            temp_table.grids
                = value_grid_ids.insert_back(grid_ids.begin(), grid_ids.end());
            temp_table_ids[mat_idx] = value_tables.push_back(temp_table);
        }
        // Construct cross section table for this model
        ModelXsTable temp_model_xs;
        temp_model_xs.material = value_table_ids.insert_back(
            temp_table_ids.begin(), temp_table_ids.end());
        model_xs.push_back(temp_model_xs);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct energy loss fluctuation model data.
 */
void PhysicsParams::build_fluct(const Options&        opts,
                                const MaterialParams& mats,
                                const ParticleParams& particles,
                                HostValue*            data) const
{
    CELER_EXPECT(data);

    if (!opts.enable_fluctuation)
        return;

    // Set particle properties
    data->fluctuation.electron_id = particles.find(pdg::electron());
    CELER_VALIDATE(data->fluctuation.electron_id,
                   << "missing electron particle (required for energy loss "
                      "fluctuations)");
    data->fluctuation.electron_mass
        = particles.get(data->fluctuation.electron_id).mass().value();

    // Loop over materials
    for (auto mat_id : range(MaterialId{mats.size()}))
    {
        const auto mat = mats.get(mat_id);

        // Calculate the parameters for the energy loss fluctuation model (see
        // Geant3 PHYS332 2.4 and Geant4 physics reference manual 7.3.2)
        UrbanFluctuationParameters params;
        const real_type avg_z = mat.electron_density() / mat.number_density();
        params.oscillator_strength[1] = avg_z > 2 ? 2 / avg_z : 0;
        params.oscillator_strength[0] = 1 - params.oscillator_strength[1];
        params.binding_energy[1]      = 1e-5 * ipow<2>(avg_z);
        params.binding_energy[0]
            = std::pow(mat.mean_excitation_energy().value()
                           / std::pow(params.binding_energy[1],
                                      params.oscillator_strength[1]),
                       1 / params.oscillator_strength[0]);
        params.log_binding_energy[1] = std::log(params.binding_energy[1]);
        params.log_binding_energy[0] = std::log(params.binding_energy[0]);
        make_builder(&data->fluctuation.urban).push_back(params);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas

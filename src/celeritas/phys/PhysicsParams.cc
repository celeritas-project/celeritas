//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/model/CombinedBremModel.hh"
#include "celeritas/em/model/EPlusGGModel.hh"
#include "celeritas/em/model/LivermorePEModel.hh"
#include "celeritas/em/params/AtomicRelaxationParams.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/grid/RangeGridCalculator.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/grid/XsGridInserter.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"
#include "celeritas/neutron/model/ChipsNeutronElasticModel.hh"

#include "Model.hh"
#include "ParticleParams.hh"
#include "PhysicsData.hh"
#include "Process.hh"

#include "detail/DiscreteSelectAction.hh"
#include "detail/PreStepAction.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
using Values
    = Collection<real_type, Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
class ImplicitPhysicsAction final : public StaticConcreteAction
{
  public:
    // Construct with ID and label
    using StaticConcreteAction::StaticConcreteAction;
};

//---------------------------------------------------------------------------//
//! PDG recommends 81-100 for internal MC pseudoparticles
bool is_fake_particle(PDGNumber pdg)
{
    return pdg.get() >= 81 && pdg.get() <= 100;
}

//---------------------------------------------------------------------------//
//! Calculate the energy of the maximum cross section.
real_type calc_energy_max_xs(UniformGridRecord const& data, Values const& reals)
{
    UniformGrid loge_grid(data.grid);
    UniformLogGridCalculator calc_xs(data, reals);

    real_type xs_max = 0;
    real_type result = 0;
    for (auto i : range(loge_grid.size()))
    {
        real_type xs = calc_xs[i];
        if (xs > xs_max)
        {
            xs_max = xs;
            result = std::exp(loge_grid[i]);
        }
    }
    CELER_ENSURE(result > 0);
    return result;
}

//---------------------------------------------------------------------------//
//! Calculate the energy of the maximum cross section.
real_type calc_energy_max_xs(XsGridRecord const& data, Values const& reals)
{
    CELER_EXPECT(data);

    real_type result{0};
    if (data.lower)
    {
        result = calc_energy_max_xs(data.lower, reals);
    }
    if (data.upper)
    {
        result = max(result, calc_energy_max_xs(data.upper, reals));
    }
    CELER_ENSURE(result > 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with processes and helper classes.
 */
PhysicsParams::PhysicsParams(Input inp)
    : processes_(std::move(inp.processes))
    , relaxation_(std::move(inp.relaxation))
{
    CELER_EXPECT(!processes_.empty());
    CELER_EXPECT(
        std::all_of(processes_.begin(), processes_.end(), LogicalTrue{}));
    CELER_EXPECT(inp.particles);
    CELER_EXPECT(inp.materials);
    CELER_EXPECT(inp.action_registry);

    ScopedMem record_mem("PhysicsParams.construct");

    // Create actions (order matters due to accessors in PhysicsParamsScalars)
    {
        using std::make_shared;
        auto& action_reg = *inp.action_registry;

        auto pre_step_action
            = make_shared<detail::PreStepAction>(action_reg.next_id());
        inp.action_registry->insert(pre_step_action);
        pre_step_action_ = std::move(pre_step_action);

        auto msc_action = make_shared<ImplicitPhysicsAction>(
            action_reg.next_id(),
            "msc-range",
            "limit range due to multiple scattering");
        action_reg.insert(msc_action);
        msc_action_ = std::move(msc_action);

        auto range_action = make_shared<ImplicitPhysicsAction>(
            action_reg.next_id(),
            "eloss-range",
            "limit range due to energy loss");
        action_reg.insert(range_action);
        range_action_ = std::move(range_action);

        auto discrete_action
            = make_shared<detail::DiscreteSelectAction>(action_reg.next_id());
        inp.action_registry->insert(discrete_action);
        discrete_action_ = std::move(discrete_action);

        auto integral_action = make_shared<ImplicitPhysicsAction>(
            action_reg.next_id(),
            "physics-integral-rejected",
            "reject by integral cross section");
        inp.action_registry->insert(integral_action);
        integral_rejection_action_ = std::move(integral_action);

        // Emit models for associated proceses
        models_ = this->build_models(inp.action_registry);

        // Place "failure" *after* all the model IDs
        auto failure_action = make_shared<ImplicitPhysicsAction>(
            action_reg.next_id(),
            "physics-failure",
            "mark a track that failed to sample an interaction");
        inp.action_registry->insert(failure_action);
        failure_action_ = std::move(failure_action);
    }

    // Construct data
    HostValue host_data;
    this->build_options(inp.options, &host_data);
    this->build_ids(*inp.particles, &host_data);
    this->build_tables(inp.options, *inp.materials, &host_data);
    this->build_model_tables(*inp.materials, &host_data);

    // Add step limiter if being used (TODO: remove this hack from physics)
    if (inp.options.fixed_step_limiter > 0)
    {
        using std::make_shared;
        auto& action_reg = *inp.action_registry;

        auto fixed_step_action = make_shared<ImplicitPhysicsAction>(
            action_reg.next_id(),
            "physics-fixed-step",
            "fixed step limiter for charged particles");
        inp.action_registry->insert(fixed_step_action);
        host_data.scalars.fixed_step_limiter = inp.options.fixed_step_limiter;
        host_data.scalars.fixed_step_action = fixed_step_action->action_id();
        fixed_step_action_ = std::move(fixed_step_action);
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
    CELER_EXPECT(id < this->host_ref().process_groups.size());
    auto const& data = this->host_ref();
    return data.process_ids[data.process_groups[id].processes];
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
auto PhysicsParams::build_models(ActionRegistry* mgr) const -> VecModel
{
    VecModel models;

    // Construct models, assigning each model ID
    for (auto process_idx : range<ProcessId::size_type>(processes_.size()))
    {
        auto id_iter = Process::ActionIdIter{mgr->next_id()};
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
 * Construct on-device particle-dependent physics options.
 */
void PhysicsParams::build_particle_options(ParticleOptions const& opts,
                                           ParticleScalars* data) const
{
    CELER_VALIDATE(opts.min_range > 0,
                   << "invalid min_range=" << opts.min_range
                   << " (should be positive)");
    CELER_VALIDATE(opts.max_step_over_range > 0,
                   << "invalid max_step_over_range="
                   << opts.max_step_over_range << " (should be positive)");
    CELER_VALIDATE(opts.lowest_energy.value() > 0,
                   << "invalid lowest_energy=" << opts.lowest_energy.value()
                   << " (should be positive)");
    CELER_VALIDATE(opts.range_factor > 0 && opts.range_factor < 1,
                   << "invalid range_factor=" << opts.range_factor
                   << " (should be within 0 < limit < 1)");
    data->min_range = opts.min_range;
    data->max_step_over_range = opts.max_step_over_range;
    data->lowest_energy = opts.lowest_energy;
    data->displaced = opts.displaced;
    data->range_factor = opts.range_factor;
    data->step_limit_algorithm = opts.step_limit_algorithm;
    if (data->step_limit_algorithm
        == MscStepLimitAlgorithm::distance_to_boundary)
    {
        CELER_LOG(warning) << "Unsupported MSC step limit algorithm '"
                           << data->step_limit_algorithm
                           << "': defaulting to '"
                           << MscStepLimitAlgorithm::safety << "'";
        data->step_limit_algorithm = MscStepLimitAlgorithm::safety;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct on-device physics options.
 */
void PhysicsParams::build_options(Options const& opts, HostValue* data) const
{
    CELER_VALIDATE(opts.min_eprime_over_e > 0 && opts.min_eprime_over_e < 1,
                   << "invalid min_eprime_over_e=" << opts.min_eprime_over_e
                   << " (should be be within 0 < limit < 1)");
    CELER_VALIDATE(opts.linear_loss_limit >= 0 && opts.linear_loss_limit <= 1,
                   << "invalid linear_loss_limit=" << opts.linear_loss_limit
                   << " (should be within 0 <= limit <= 1)");
    CELER_VALIDATE(opts.secondary_stack_factor > 0,
                   << "invalid secondary_stack_factor="
                   << opts.secondary_stack_factor << " (should be positive)");
    CELER_VALIDATE(opts.lambda_limit > 0,
                   << "invalid lambda_limit=" << opts.lambda_limit
                   << " (should be positive)");
    CELER_VALIDATE(opts.safety_factor >= 0.1,
                   << "invalid safety_factor=" << opts.safety_factor
                   << " (should be >= 0.1)");
    data->scalars.min_eprime_over_e = opts.min_eprime_over_e;
    data->scalars.linear_loss_limit = opts.linear_loss_limit;
    data->scalars.secondary_stack_factor = opts.secondary_stack_factor;
    data->scalars.lambda_limit = opts.lambda_limit;
    data->scalars.safety_factor = opts.safety_factor;

    this->build_particle_options(opts.light, &data->scalars.light);
    this->build_particle_options(opts.heavy, &data->scalars.heavy);
}

//---------------------------------------------------------------------------//
/*!
 * Construct particle -> process -> model mappings.
 */
void PhysicsParams::build_ids(ParticleParams const& particles,
                              HostValue* data) const
{
    CELER_EXPECT(data);
    CELER_EXPECT(!models_.empty());
    using ModelRange = std::tuple<real_type, real_type, ParticleModelId>;

    // Offset from the index in the list of models to a model's ActionId
    data->scalars.model_to_action = this->model(ModelId{0})->action_id().get();

    // Note: use map to keep ProcessId sorted
    std::vector<std::map<ProcessId, std::vector<ModelRange>>> particle_models(
        particles.size());
    std::vector<ModelId> temp_model_ids;
    ParticleModelId::size_type pm_idx{0};

    // Construct particle -> process -> model map
    for (auto mid : range(ModelId{this->num_models()}))
    {
        Model const& m = *this->model(mid);
        ProcessId const process_id = this->process_id(mid);
        for (Applicability const& applic : m.applicability())
        {
            if (applic.material)
            {
                CELER_NOT_IMPLEMENTED("material-dependent models");
            }
            CELER_VALIDATE(applic.particle < particles.size(),
                           << "invalid particle ID "
                           << applic.particle.unchecked_get());
            CELER_VALIDATE(applic.lower < applic.upper,
                           << "expected lower energy limit ("
                           << value_as<ModelGroup::Energy>(applic.lower)
                           << " MeV) to be less than upper energy limit ("
                           << value_as<ModelGroup::Energy>(applic.upper)
                           << " MeV) for model " << m.label());
            particle_models[applic.particle.get()][process_id].push_back(
                {value_as<ModelGroup::Energy>(applic.lower),
                 value_as<ModelGroup::Energy>(applic.upper),
                 ParticleModelId{pm_idx++}});
            temp_model_ids.push_back(mid);
        }
    }
    make_builder(&data->model_ids)
        .insert_back(temp_model_ids.begin(), temp_model_ids.end());

    auto process_groups = make_builder(&data->process_groups);
    auto process_ids = make_builder(&data->process_ids);
    auto model_groups = make_builder(&data->model_groups);
    auto pmodel_ids = make_builder(&data->pmodel_ids);
    auto reals = make_builder(&data->reals);

    process_groups.reserve(particle_models.size());

    // Loop over particle IDs, set ProcessGroup
    ProcessId::size_type max_particle_processes = 0;
    for (auto par_id : range(ParticleId{particles.size()}))
    {
        auto& process_to_models = particle_models[par_id.get()];
        if (process_to_models.empty()
            && !is_fake_particle(particles.id_to_pdg(par_id)))
        {
            CELER_LOG(warning) << "No processes are defined for particle '"
                               << particles.id_to_label(par_id) << '\'';
        }
        max_particle_processes = std::max<ProcessId::size_type>(
            max_particle_processes, process_to_models.size());

        std::vector<ProcessId> temp_processes;
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
            std::vector<real_type> temp_energy_grid;
            std::vector<ParticleModelId> temp_models;
            temp_energy_grid.reserve(models.size() + 1);
            temp_models.reserve(models.size());

            // Sort, and add the first grid point
            std::sort(models.begin(), models.end());
            temp_energy_grid.push_back(std::get<0>(models[0]));

            for (ModelRange const& r : models)
            {
                CELER_VALIDATE(
                    temp_energy_grid.back() == std::get<0>(r),
                    << "models for process '"
                    << this->process(pid_models.first)->label()
                    << "' of particle type '" << particles.id_to_label(par_id)
                    << "' has no data between energies of "
                    << temp_energy_grid.back() << " and " << std::get<0>(r)
                    << " (energy range must be contiguous)");
                temp_energy_grid.push_back(std::get<1>(r));
                temp_models.push_back(std::get<2>(r));
            }

            ModelGroup mdata;
            mdata.energy = reals.insert_back(temp_energy_grid.begin(),
                                             temp_energy_grid.end());
            mdata.model = pmodel_ids.insert_back(temp_models.begin(),
                                                 temp_models.end());
            CELER_ASSERT(mdata);
            temp_model_datas.push_back(mdata);
        }

        ProcessGroup pdata;
        pdata.processes = process_ids.insert_back(temp_processes.begin(),
                                                  temp_processes.end());
        pdata.models = model_groups.insert_back(temp_model_datas.begin(),
                                                temp_model_datas.end());

        // It's ok to have particles defined in the problem that do not have
        // any processes (if they are ever created, they will just be
        // transported until they exit the geometry).
        // NOTE: data tables will be assigned later
        CELER_ASSERT(process_to_models.empty() || pdata);
        process_groups.push_back(pdata);
    }
    data->scalars.max_particle_processes = max_particle_processes;
    data->scalars.num_models = this->num_models();

    // Assign hardwired models that do on-the-fly xs calculation
    for (auto model_idx : range(this->num_models()))
    {
        Model const& model = *models_[model_idx].first;
        ProcessId const process_id = models_[model_idx].second;
        if (auto* pe_model = dynamic_cast<LivermorePEModel const*>(&model))
        {
            data->hardwired.photoelectric = process_id;
            data->hardwired.photoelectric_table_thresh = units::MevEnergy{0.2};
            data->hardwired.livermore_pe = ModelId{model_idx};
            data->hardwired.livermore_pe_data = pe_model->host_ref();
        }
        else if (auto* epgg_model = dynamic_cast<EPlusGGModel const*>(&model))
        {
            data->hardwired.positron_annihilation = process_id;
            data->hardwired.eplusgg = ModelId{model_idx};
            data->hardwired.eplusgg_data = epgg_model->device_ref();
        }
        else if (auto* ne_model
                 = dynamic_cast<ChipsNeutronElasticModel const*>(&model))
        {
            data->hardwired.neutron_elastic = process_id;
            data->hardwired.chips = ModelId{model_idx};
            data->hardwired.chips_data = ne_model->device_ref();
        }
    }

    if (relaxation_)
    {
        // TODO: this makes a copy of all the data rather than a
        // host/device reference
        data->hardwired.relaxation_data = relaxation_->host_ref();
    }

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
/*!
 * Construct cross section data.
 */
void PhysicsParams::build_tables(Options const& opts,
                                 MaterialParams const& mats,
                                 HostValue* data) const
{
    CELER_EXPECT(*data);

    using Energy = Applicability::Energy;

    DedupeCollectionBuilder reals(&data->reals);
    CollectionBuilder grid_ids(&data->value_grid_ids);
    CollectionBuilder grids(&data->value_grids);
    CollectionBuilder tables(&data->value_tables);
    CollectionBuilder integral_xs(&data->integral_xs);

    XsGridInserter insert_grid(&data->reals, &data->value_grids);
    auto build_grid = [&insert_grid](inp::XsGrid const& grid) -> ValueGridId {
        return grid ? insert_grid(grid) : ValueGridId{};
    };

    Applicability applic;
    for (auto particle_id : range(ParticleId(data->process_groups.size())))
    {
        applic.particle = particle_id;

        // Processes for this particle
        ProcessGroup& process_group = data->process_groups[particle_id];
        auto process_ids = data->process_ids[process_group.processes];
        auto model_groups = data->model_groups[process_group.models];
        CELER_ASSERT(process_ids.size() == model_groups.size());

        // Material-dependent physics tables, one cross section table per
        // particle-process and one dedx/range table per particle
        std::vector<ValueTable> temp_macro_xs(process_ids.size());
        ValueTable temp_energy_loss;
        ValueTable temp_range;
        ValueTable temp_inv_range;

        // Processes with dE/dx and macro xs tables
        std::vector<IntegralXsProcess> temp_integral_xs(process_ids.size());

        // Loop over per-particle processes
        for (auto pp_idx : range(process_ids.size()))
        {
            // Get energy bounds for this process
            auto energy_grid = data->reals[model_groups[pp_idx].energy];
            applic.lower = Energy{energy_grid.front()};
            applic.upper = Energy{energy_grid.back()};
            CELER_ASSERT(applic.lower < applic.upper);

            Process const& proc = *this->process(process_ids[pp_idx]);

            // Grid IDs for each grid type, each material
            std::vector<ValueGridId> macro_xs_ids(mats.size());
            std::vector<ValueGridId> energy_loss_ids(mats.size());
            std::vector<ValueGridId> range_ids(mats.size(), ValueGridId{});
            std::vector<ValueGridId> inv_range_ids;

            // Energy of maximum cross section for each material
            std::vector<real_type> energy_max_xs;
            bool use_integral_xs = !opts.disable_integral_xs
                                   && proc.supports_integral_xs();
            if (use_integral_xs)
            {
                energy_max_xs.resize(mats.size());
            }

            if (proc.applies_at_rest())
            {
                /* \todo For now assume only one process per particle applies
                 * at rest. If a particle has multiple at-rest processes, we
                 * will need to check which process has the shortest time to
                 * interaction and choose that process in \c
                 * select_discrete_interaction.
                 */
                CELER_VALIDATE(!process_group.at_rest,
                               << "particle ID " << particle_id.get()
                               << " has multiple at-rest processes");

                // Discrete interaction can occur at rest
                process_group.at_rest = ParticleProcessId(pp_idx);
            }

            // Loop over materials
            for (auto mat_idx : range(mats.size()))
            {
                applic.material = PhysMatId(mat_idx);

                // Construct macroscopic cross section grid
                auto macro_xs = proc.macro_xs(applic);
                macro_xs_ids[mat_idx] = build_grid(macro_xs);

                // Construct energy loss grid
                auto energy_loss = proc.energy_loss(applic);
                energy_loss_ids[mat_idx] = build_grid(energy_loss);

                // Construct range grid from energy loss
                if (energy_loss)
                {
                    auto range
                        = RangeGridCalculator(BC::geant)(energy_loss.lower);
                    range_ids[mat_idx] = insert_grid({range, {}});

                    if (range.interpolation.type
                        == InterpolationType::cubic_spline)
                    {
                        // Build the inverse range grid if cubic spline
                        // interpolation is used
                        inv_range_ids.resize(mats.size(), ValueGridId{});

                        // The range and energy values are not inverted on the
                        // grid, but the derivatives are calculated using the
                        // inverted grid.
                        auto inv_range = data->value_grids[range_ids[mat_idx]];
                        auto derivative
                            = SplineDerivCalculator(BC::geant).calc_from_inverse(
                                inv_range.lower, make_const_ref(*data).reals);
                        inv_range.lower.derivative = reals.insert_back(
                            derivative.begin(), derivative.end());
                        inv_range_ids[mat_idx] = grids.push_back(inv_range);
                    }
                }

                if (use_integral_xs)
                {
                    // Find and store the energy of the largest cross section
                    // for this material if the integral approach is used

                    if (process_ids[pp_idx]
                        == data->hardwired.positron_annihilation)
                    {
                        // Annihilation cross section is maximum at zero and
                        // decreases with increasing energy
                        energy_max_xs[mat_idx] = 0;
                    }
                    else if (auto grid_id = macro_xs_ids[mat_idx])
                    {
                        energy_max_xs[mat_idx]
                            = calc_energy_max_xs(data->value_grids[grid_id],
                                                 make_const_ref(*data).reals);
                    }
                }
            }

            // Check if any material has value grids
            auto has_grids = [](std::vector<ValueGridId> const& v) {
                return std::any_of(v.begin(), v.end(), [](ValueGridId id) {
                    return static_cast<bool>(id);
                });
            };

            // Construct value grid tables
            if (has_grids(macro_xs_ids))
            {
                temp_macro_xs[pp_idx].grids = grid_ids.insert_back(
                    macro_xs_ids.begin(), macro_xs_ids.end());
                CELER_ASSERT(temp_macro_xs[pp_idx].grids.size() == mats.size());
            }
            if (has_grids(energy_loss_ids))
            {
                CELER_ASSERT(has_grids(range_ids));
                CELER_VALIDATE(!temp_energy_loss && !temp_range,
                               << "more than one process for particle ID "
                               << particle_id.get()
                               << " has energy loss tables");

                temp_energy_loss.grids = grid_ids.insert_back(
                    energy_loss_ids.begin(), energy_loss_ids.end());
                CELER_ASSERT(temp_energy_loss.grids.size() == mats.size());

                temp_range.grids
                    = grid_ids.insert_back(range_ids.begin(), range_ids.end());
                CELER_ASSERT(temp_range.grids.size() == mats.size());

                temp_inv_range.grids = grid_ids.insert_back(
                    inv_range_ids.begin(), inv_range_ids.end());
                CELER_ASSERT(temp_inv_range.grids.size() == mats.size()
                             || temp_inv_range.grids.empty());
            }

            // Store the energies of the maximum cross sections
            if (!energy_max_xs.empty())
            {
                temp_integral_xs[pp_idx].energy_max_xs = reals.insert_back(
                    energy_max_xs.begin(), energy_max_xs.end());
            }
        }
        // Construct energy loss process data
        process_group.integral_xs = integral_xs.insert_back(
            temp_integral_xs.begin(), temp_integral_xs.end());

        // Construct value tables
        process_group.macro_xs
            = tables.insert_back(temp_macro_xs.begin(), temp_macro_xs.end());
        process_group.energy_loss = tables.push_back(temp_energy_loss);
        process_group.range = tables.push_back(temp_range);
        process_group.inverse_range = tables.push_back(temp_inv_range);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct model cross section CDFs.
 */
void PhysicsParams::build_model_tables(MaterialParams const& mats,
                                       HostValue* data) const
{
    CELER_EXPECT(*data);

    XsGridInserter insert(&data->reals, &data->value_grids);
    CollectionBuilder model_cdf(&data->model_cdf);
    CollectionBuilder tables(&data->value_tables);
    CollectionBuilder table_ids(&data->value_table_ids);
    CollectionBuilder grid_ids(&data->value_grid_ids);

    for (auto model_idx : range(this->num_models()))
    {
        Model const& model = *models_[model_idx].first;

        // Loop over applicable particles
        for (Applicability applic : model.applicability())
        {
            std::vector<ValueTableId> temp_table_ids(data->model_ids.size());
            for (auto mat_id : range(PhysMatId{mats.size()}))
            {
                // Construct microscopic cross sections
                applic.material = mat_id;
                auto elements = mats.get(mat_id).elements();
                auto grids = model.micro_xs(applic);

                if (grids.empty() || elements.size() == 1)
                {
                    // Models with material-independent discrete interactions
                    // or on-the-fly xs calculation won't have micro xs grids
                    continue;
                }

                // Get the number of grid points: the energy grids are the
                // same for each element in the material
                CELER_ASSERT(grids.size() == elements.size());
                CELER_ASSERT(grids.front().lower && !grids.front().upper);

                // Calculate the cross section CDF
                for (auto i : range(grids.front().lower.y.size()))
                {
                    double accum{0};
                    for (auto j : range(elements.size()))
                    {
                        CELER_ASSERT(grids[j].lower);
                        auto& value = grids[j].lower.y[i];
                        accum += value * elements[j].fraction;
                        value = accum;
                    }
                    if (accum > 0)
                    {
                        // Normalize the CDF
                        double norm = 1 / accum;
                        for (auto j : range(elements.size() - 1))
                        {
                            grids[j].lower.y[i] *= norm;
                        }
                        grids.back().lower.y[i] = 1;
                    }
                }

                // Construct grids for each element in the material
                std::vector<ValueGridId> temp_grid_ids(elements.size());
                for (auto elcomp_idx : range(elements.size()))
                {
                    temp_grid_ids[elcomp_idx] = insert(grids[elcomp_idx]);
                }

                // Construct table for the material
                ValueTable table;
                table.grids = grid_ids.insert_back(temp_grid_ids.begin(),
                                                   temp_grid_ids.end());
                temp_table_ids[mat_id.get()] = tables.push_back(table);
            }
            // Construct table for the model
            ModelCdfTable cdf;
            cdf.material = table_ids.insert_back(temp_table_ids.begin(),
                                                 temp_table_ids.end());
            model_cdf.push_back(cdf);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

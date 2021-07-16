//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include <algorithm>
#include <map>
#include <tuple>
#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/VectorUtils.hh"
#include "comm/Logger.hh"
#include "ParticleParams.hh"
#include "physics/em/EPlusGGModel.hh"
#include "physics/em/LivermorePEModel.hh"
#include "physics/grid/ValueGridInserter.hh"
#include "physics/grid/XsCalculator.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with processes and helper classes.
 */
PhysicsParams::PhysicsParams(Input inp) : processes_(std::move(inp.processes))
{
    CELER_EXPECT(!processes_.empty());
    CELER_EXPECT(std::all_of(processes_.begin(),
                             processes_.end(),
                             [](const SPConstProcess& p) { return bool(p); }));
    CELER_EXPECT(inp.particles);
    CELER_EXPECT(inp.materials);

    // Emit models for associated proceses
    models_ = this->build_models();

    // Construct data
    HostValue host_data;
    this->build_options(inp.options, &host_data);
    this->build_ids(*inp.particles, &host_data);
    this->build_xs(inp.options, *inp.materials, &host_data);

    CELER_LOG(debug)
        << "Constructed physics sizes:"
        << "\n  reals: " << host_data.reals.size()
        << "\n  model_ids: " << host_data.model_ids.size()
        << "\n  value_grids: " << host_data.value_grids.size()
        << "\n  value_grid_ids: " << host_data.value_grid_ids.size()
        << "\n  process_ids: " << host_data.process_ids.size()
        << "\n  value_tables: " << host_data.value_tables.size()
        << "\n  model_groups: " << host_data.model_groups.size()
        << "\n  process_groups: " << host_data.process_groups.size();

    data_ = CollectionMirror<PhysicsParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the list of process IDs that apply to a particle type.
 */
auto PhysicsParams::processes(ParticleId id) const -> SpanConstProcessId
{
    CELER_EXPECT(id < this->num_processes());
    const auto& data = this->host_pointers();
    return data.process_ids[data.process_groups[id].processes];
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
auto PhysicsParams::build_models() const -> VecModel
{
    VecModel models;

    // Construct models, assigning each model ID
    ModelIdGenerator next_model_id;
    for (auto process_idx : range<ProcessId::size_type>(processes_.size()))
    {
        auto new_models = processes_[process_idx]->build_models(next_model_id);
        CELER_ASSERT(!new_models.empty());
        for (SPConstModel& model : new_models)
        {
            CELER_ASSERT(model);
            ModelId model_id = next_model_id();
            CELER_ASSERT(model->model_id() == model_id);

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
                   << " (should be be within 0 <= limit <= 1)");
    data->scaling_min_range  = opts.min_range;
    data->scaling_fraction   = opts.max_step_over_range;
    data->energy_fraction    = opts.min_eprime_over_e;
    data->linear_loss_limit  = opts.linear_loss_limit;
    data->enable_eloss_fluct = opts.enable_eloss_fluct;
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
    using ModelRange = std::tuple<real_type, real_type, ModelId>;

    // Note: use map to keep ProcessId sorted
    std::vector<std::map<ProcessId, std::vector<ModelRange>>> particle_models(
        particles.size());

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
                {applic.lower.value(),
                 applic.upper.value(),
                 ModelId{model_idx}});
        }
    }

    auto process_groups = make_builder(&data->process_groups);
    auto process_ids    = make_builder(&data->process_ids);
    auto model_groups   = make_builder(&data->model_groups);
    auto model_ids      = make_builder(&data->model_ids);
    auto reals          = make_builder(&data->reals);

    process_groups.reserve(particle_models.size());

    // Loop over particle IDs, set ProcessGroup
    for (auto particle_idx : range(particles.size()))
    {
        auto& process_to_models = particle_models[particle_idx];
        if (process_to_models.empty())
        {
            CELER_LOG(warning)
                << "No processes are defined for particle '"
                << particles.id_to_label(ParticleId{particle_idx}) << '\'';
        }
        data->max_particle_processes = std::max<ProcessId::size_type>(
            data->max_particle_processes, process_to_models.size());

        std::vector<ProcessId>  temp_processes;
        std::vector<ModelGroup> temp_model_groups;
        temp_processes.reserve(process_to_models.size());
        temp_model_groups.reserve(process_to_models.size());
        for (auto& pid_models : process_to_models)
        {
            // Add process ID
            temp_processes.push_back(pid_models.first);

            std::vector<ModelRange>& models = pid_models.second;
            CELER_ASSERT(!models.empty());

            // Construct model group
            std::vector<real_type> temp_energy_grid;
            std::vector<ModelId>   temp_models;
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

            ModelGroup mgroup;
            mgroup.energy = reals.insert_back(temp_energy_grid.begin(),
                                              temp_energy_grid.end());
            mgroup.model  = model_ids.insert_back(temp_models.begin(),
                                                 temp_models.end());
            CELER_ASSERT(mgroup);
            temp_model_groups.push_back(mgroup);
        }

        ProcessGroup pgroup;
        pgroup.processes = process_ids.insert_back(temp_processes.begin(),
                                                   temp_processes.end());
        pgroup.models    = model_groups.insert_back(temp_model_groups.begin(),
                                                 temp_model_groups.end());

        // It's ok to have particles defined in the problem that do not have
        // any processes (if they are ever created, they will just be
        // transported until they exit the geometry).
        // NOTE: data tables will be assigned later
        CELER_ASSERT(process_to_models.empty() || pgroup);
        process_groups.push_back(pgroup);
    }

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
            data->hardwired.livermore_pe_data = pe_model->host_pointers();
        }
        else if (auto* epgg_model = dynamic_cast<const EPlusGGModel*>(&model))
        {
            data->hardwired.positron_annihilation = process_id;
            data->hardwired.eplusgg               = ModelId{model_idx};
            data->hardwired.eplusgg_params = epgg_model->device_pointers();
        }
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
        ProcessGroup& process_group = data->process_groups[particle_id];
        Span<const ProcessId> processes
            = data->process_ids[process_group.processes];
        Span<const ModelGroup> model_groups
            = data->model_groups[process_group.models];
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
            applic.lower = Applicability::Energy{energy_grid.front()};
            applic.upper = Applicability::Energy{energy_grid.back()};
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

                // If this is an energy loss process, find and store the energy
                // of the largest cross section for this material
                auto grid_id
                    = temp_grid_ids[ValueGridType::macro_xs][mat_id.get()];
                if (proc.type() == ProcessType::electromagnetic_dedx
                    && opts.use_integral_xs && grid_id)
                {
                    // Allocate storage for the energies if we haven't already
                    energy_max_xs.resize(mats.size());

                    const auto&        grid_data = data->value_grids[grid_id];
                    auto               data_ref  = make_const_ref(*data);
                    const UniformGrid  loge_grid(grid_data.log_energy);
                    const XsCalculator calc_xs(grid_data, data_ref.reals);

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

                // Index of the energy loss process that stores the de/dx and
                // range tables
                if (temp_grid_ids[ValueGridType::energy_loss][mat_id.get()]
                    && temp_grid_ids[ValueGridType::range][mat_id.get()])
                {
                    // Only one particle-process should have energy loss tables
                    CELER_ASSERT(!process_group.eloss_ppid
                                 || pp_idx == process_group.eloss_ppid.get());
                    process_group.eloss_ppid = ParticleProcessId{pp_idx};
                }
            }

            // Outer loop over grid types
            for (auto vgt : range(ValueGridType::size_))
            {
                if (!std::any_of(temp_grid_ids[vgt].begin(),
                                 temp_grid_ids[vgt].end(),
                                 [](ValueGridId id) { return bool(id); }))
                {
                    // Skip this table type since it's not present for any
                    // material for this particle process
                    CELER_LOG(debug) << "No " << to_cstring(ValueGridType(vgt))
                                     << " for process " << proc.label();
                    continue;
                }

                // Construct value grid table
                ValueTable& temp_table = temp_tables[vgt][pp_idx];
                temp_table.material    = value_grid_ids.insert_back(
                    temp_grid_ids[vgt].begin(), temp_grid_ids[vgt].end());
                CELER_ASSERT(temp_table.material.size() == mats.size());
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
        process_group.integral_xs = integral_xs.insert_back(
            temp_integral_xs.begin(), temp_integral_xs.end());

        // Construct value tables
        for (auto vgt : range(ValueGridType::size_))
        {
            process_group.tables[vgt] = value_tables.insert_back(
                temp_tables[vgt].begin(), temp_tables[vgt].end());
        }
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas

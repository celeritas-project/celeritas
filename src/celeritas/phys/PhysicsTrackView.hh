//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/xs/EPlusGGMacroXsCalculator.hh"
#include "celeritas/em/xs/LivermorePEMacroXsCalculator.hh"
#include "celeritas/grid/GridIdFinder.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/mat/TabulatedElementSelector.hh"

#include "PhysicsData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics data for a track.
 *
 * The physics track view provides an interface for data and operations
 * common to most processes and models.
 */
class PhysicsTrackView
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t = PhysicsTrackInitializer;
    using FluctuationRef
        = FluctuationData<Ownership::const_reference, MemSpace::native>;
    using PhysicsParamsRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::native>;
    using PhysicsStateRef
        = PhysicsStateData<Ownership::reference, MemSpace::native>;
    using UrbanMscRef
        = UrbanMscData<Ownership::const_reference, MemSpace::native>;
    using Energy      = units::MevEnergy;
    using ModelFinder = GridIdFinder<Energy, ParticleModelId>;
    //!@}

  public:
    // Construct from params, states, and per-state IDs
    inline CELER_FUNCTION PhysicsTrackView(const PhysicsParamsRef& params,
                                           const PhysicsStateRef&  states,
                                           ParticleId              particle,
                                           MaterialId              material,
                                           ThreadId                id);

    // Initialize the track view
    inline CELER_FUNCTION PhysicsTrackView& operator=(const Initializer_t&);

    // Set the remaining MFP to interaction
    inline CELER_FUNCTION void interaction_mfp(real_type);

    // Reset the remaining MFP to interaction
    inline CELER_FUNCTION void reset_interaction_mfp();

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Whether the remaining MFP has been calculated
    CELER_FORCEINLINE_FUNCTION bool has_interaction_mfp() const;

    // Remaining MFP to interaction [1]
    CELER_FORCEINLINE_FUNCTION real_type interaction_mfp() const;

    //// PROCESSES (depend on particle type and possibly material) ////

    // Number of processes that apply to this track
    inline CELER_FUNCTION ParticleProcessId::size_type
                          num_particle_processes() const;

    // Process ID for the given within-particle process index
    inline CELER_FUNCTION ProcessId process(ParticleProcessId) const;

    // Get table, null if not present for this particle/material/type
    inline CELER_FUNCTION ValueGridId value_grid(ValueGridType table,
                                                 ParticleProcessId) const;

    // Whether to use integral approach to sample the discrete interaction
    inline CELER_FUNCTION bool use_integral_xs(ParticleProcessId ppid) const;

    // Energy corresponding to the maximum cross section for the material
    inline CELER_FUNCTION real_type energy_max_xs(ParticleProcessId ppid) const;

    // Calculate macroscopic cross section for the process
    inline CELER_FUNCTION real_type calc_xs(ParticleProcessId ppid,
                                            ValueGridId       grid_id,
                                            Energy            energy) const;

    // Models that apply to the given process ID
    inline CELER_FUNCTION
        ModelFinder make_model_finder(ParticleProcessId) const;

    // Return value table data for the given particle/model/material
    inline CELER_FUNCTION ValueTableId value_table(ParticleModelId) const;

    // Construct an element selector
    inline CELER_FUNCTION
        TabulatedElementSelector make_element_selector(ValueTableId,
                                                       Energy) const;

    // Whether the particle can have a discrete interaction at rest
    inline CELER_FUNCTION bool has_at_rest() const;

    //// PARAMETER DATA ////

    // Convert an action to a model ID for diagnostics, empty if not a model
    inline CELER_FUNCTION ModelId action_to_model(ActionId) const;

    // Convert a selected model ID into a simulation action ID
    inline CELER_FUNCTION ActionId model_to_action(ModelId) const;

    // Get the model ID corresponding to the given ParticleModelId
    inline CELER_FUNCTION ModelId model_id(ParticleModelId) const;

    // Calculate scaled step range
    inline CELER_FUNCTION real_type range_to_step(real_type range) const;

    // Access scalar properties
    CELER_FORCEINLINE_FUNCTION const PhysicsParamsScalars& scalars() const;

    // Energy loss fluctuation model parameters
    inline CELER_FUNCTION const FluctuationRef& fluctuation() const;

    // Urban multiple scattering data
    inline CELER_FUNCTION const UrbanMscRef& urban_data() const;

    // Calculate macroscopic cross section on the fly for the given model
    inline CELER_FUNCTION real_type calc_xs_otf(ModelId             model,
                                                const MaterialView& material,
                                                Energy energy) const;

    // Number of particle types
    inline CELER_FUNCTION size_type num_particles() const;

    // Construct a grid calculator from a physics table
    template<class T>
    inline CELER_FUNCTION T make_calculator(ValueGridId) const;

    //// HACKS ////

    // Process ID for photoelectric effect
    inline CELER_FUNCTION ProcessId photoelectric_process_id() const;

    // Process ID for positron annihilation
    inline CELER_FUNCTION ProcessId eplusgg_process_id() const;

    // Get hardwired model, null if not present
    inline CELER_FUNCTION ModelId hardwired_model(ParticleProcessId ppid,
                                                  Energy energy) const;

    // Particle-process ID of the process with the de/dx and range tables
    inline CELER_FUNCTION ParticleProcessId eloss_ppid() const;

    // Particle-process ID of the process with the msc cross section table
    inline CELER_FUNCTION ParticleProcessId msc_ppid() const;

  private:
    const PhysicsParamsRef& params_;
    const PhysicsStateRef&  states_;
    const ParticleId        particle_;
    const MaterialId        material_;
    const ThreadId          thread_;

    //// IMPLEMENTATION HELPER FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState&       state();
    CELER_FORCEINLINE_FUNCTION const PhysicsTrackState& state() const;
    CELER_FORCEINLINE_FUNCTION const ProcessGroup&      process_group() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and state data.
 *
 * Particle and material IDs are derived from other class states.
 */
CELER_FUNCTION
PhysicsTrackView::PhysicsTrackView(const PhysicsParamsRef& params,
                                   const PhysicsStateRef&  states,
                                   ParticleId              pid,
                                   MaterialId              mid,
                                   ThreadId                tid)
    : params_(params)
    , states_(states)
    , particle_(pid)
    , material_(mid)
    , thread_(tid)
{
    CELER_EXPECT(thread_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the track view.
 */
CELER_FUNCTION PhysicsTrackView&
PhysicsTrackView::operator=(const Initializer_t&)
{
    this->state().interaction_mfp = 0;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Set the distance to the next interaction, in mean free paths.
 *
 * This value will be decremented at each step.
 */
CELER_FUNCTION void PhysicsTrackView::interaction_mfp(real_type count)
{
    CELER_EXPECT(count > 0);
    this->state().interaction_mfp = count;
}

//---------------------------------------------------------------------------//
/*!
 * Set the distance to the next interaction, in mean free paths.
 *
 * This value will be decremented at each step.
 */
CELER_FUNCTION void PhysicsTrackView::reset_interaction_mfp()
{
    this->state().interaction_mfp = 0;
}


//---------------------------------------------------------------------------//
/*!
 * Whether the remaining MFP has been calculated.
 */
CELER_FUNCTION bool PhysicsTrackView::has_interaction_mfp() const
{
    return this->state().interaction_mfp > 0;
}

//---------------------------------------------------------------------------//
/*!
 * Remaining MFP to interaction.
 */
CELER_FUNCTION real_type PhysicsTrackView::interaction_mfp() const
{
    real_type mfp = this->state().interaction_mfp;
    CELER_ENSURE(mfp >= 0);
    return mfp;
}

//---------------------------------------------------------------------------//
/*!
 * Number of processes that apply to this track.
 */
CELER_FUNCTION ParticleProcessId::size_type
               PhysicsTrackView::num_particle_processes() const
{
    return this->process_group().size();
}

//---------------------------------------------------------------------------//
/*!
 * Process ID for the given within-particle process index.
 */
CELER_FUNCTION ProcessId PhysicsTrackView::process(ParticleProcessId ppid) const
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    return params_.process_ids[this->process_group().processes[ppid.get()]];
}

//---------------------------------------------------------------------------//
/*!
 * Return value grid data for the given table type and process if available.
 *
 * If the result is not null, it can be used to instantiate a
 * grid Calculator.
 *
 * If the result is null, it's likely because the process doesn't have the
 * associated value (e.g. if the table type is "energy_loss" and the process is
 * not a slowing-down process).
 */
CELER_FUNCTION auto PhysicsTrackView::value_grid(ValueGridType     table_type,
                                                 ParticleProcessId ppid) const
    -> ValueGridId
{
    CELER_EXPECT(int(table_type) < int(ValueGridType::size_));
    CELER_EXPECT(ppid < this->num_particle_processes());
    ValueTableId table_id
        = this->process_group().tables[table_type][ppid.get()];

    CELER_ASSERT(table_id);
    const ValueTable& table = params_.value_tables[table_id];
    if (!table)
        return {}; // No table for this process

    CELER_EXPECT(material_ < table.grids.size());
    auto grid_id_ref = table.grids[material_.get()];
    if (!grid_id_ref)
        return {}; // No table for this particular material

    return params_.value_grid_ids[grid_id_ref];
}

//---------------------------------------------------------------------------//
/*!
 * Whether to use integral approach to sample the discrete interaction.
 *
 * For energy loss processes, the particle will have a different energy at the
 * pre- and post-step points. This means the assumption that the cross section
 * is constant along the step is no longer valid. Instead, Monte Carlo
 * integration can be used to sample the interaction for the discrete
 * process with the correct probability from the exact distribution,
 * \f[
     p = 1 - \exp \left( -\int_{E_0}^{E_1} n \sigma(E) \dif s \right),
 * \f]
 * where \f$ E_0 \f$ is the pre-step energy, \f$ E_1 \f$ is the post-step
 * energy, \em n is the atom density, and \em s is the interaction length.
 *
 * At the start of the step, the maximum value of the cross section over the
 * step \f$ \sigma_{max} \f$ is estimated and used as the macroscopic cross
 * section for the process rather than \f$ \sigma_{E_0} \f$. After the step,
 * the new value of the cross section \f$ \sigma(E_1) \f$ is calculated, and
 * the discrete interaction for the process occurs with probability
 * \f[
     p = \frac{\sigma(E_1)}{\sigma_{\max}}.
 * \f]
 *
 * See section 7.4 of the Geant4 Physics Reference (release 10.6) for details.
 */
CELER_FUNCTION bool
PhysicsTrackView::use_integral_xs(ParticleProcessId ppid) const
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    return this->energy_max_xs(ppid) > 0;
}

//---------------------------------------------------------------------------//
/*!
 * Energy corresponding to the maximum cross section for the material.
 *
 * If the \c IntegralXsProcess is "true", the integral approach is used and
 * that process has both energy loss and macro xs tables. If \c
 * energy_max_xs[material] is nonzero, both of those tables are present for
 * this material.
 */
CELER_FUNCTION real_type
PhysicsTrackView::energy_max_xs(ParticleProcessId ppid) const
{
    CELER_EXPECT(ppid < this->num_particle_processes());

    real_type                result = 0;
    const IntegralXsProcess& process
        = params_.integral_xs[this->process_group().integral_xs[ppid.get()]];
    if (process)
    {
        CELER_ASSERT(material_ < process.energy_max_xs.size());
        result = params_.reals[process.energy_max_xs[material_.get()]];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate macroscopic cross section for the process.
 *
 * If this is an energy loss process, this returns the estimate of the maximum
 * cross section over the step. If the energy of the global maximum of the
 * cross section (calculated at initialization) is in the interval \f$ [\xi
 * E_0, E_0) \f$, where \f$ E_0 \f$ is the pre-step energy and \f$ \xi \f$ is
 * \c energy_fraction (defined by default as \f$ \xi = 1 - \alpha \f$, where
 * \f$ \alpha \f$ is \c scaling_fraction),
 * \f$ \sigma_{\max} \f$ is set to the global maximum.
 * Otherwise, \f$ \sigma_{\max} = \max( \sigma(E_0), \sigma(\xi E_0) ) \f$. If
 * the cross section is not monotonic in the interval \f$ [\xi E_0, E_0) \f$
 * and the interval does not contain the global maximum, the post-step cross
 * section \f$ \sigma(E_1) \f$ may be larger than \f$ \sigma_{\max} \f$.
 */
CELER_FUNCTION real_type PhysicsTrackView::calc_xs(ParticleProcessId ppid,
                                                   ValueGridId       grid_id,
                                                   Energy energy) const
{
    auto calc_xs = this->make_calculator<XsCalculator>(grid_id);

    // Check if the integral approach is used (true if \c energy_max_xs > 0).
    // If so, an estimate of the maximum cross section over the step is used as
    // the macro xs for this process
    real_type energy_max_xs = this->energy_max_xs(ppid);
    if (energy_max_xs > 0)
    {
        real_type energy_xi = energy.value() * params_.scalars.energy_fraction;
        if (energy_max_xs >= energy_xi && energy_max_xs < energy.value())
            return calc_xs(Energy{energy_max_xs});
        return max(calc_xs(energy), calc_xs(Energy{energy_xi}));
    }

    return calc_xs(energy);
}

//---------------------------------------------------------------------------//
/*!
 * Return the model ID that applies to the given process ID and energy if the
 * process is hardwired to calculate macroscopic cross sections on the fly. If
 * the result is null, tables should be used for this process/energy.
 */
CELER_FUNCTION ModelId PhysicsTrackView::hardwired_model(ParticleProcessId ppid,
                                                         Energy energy) const
{
    ProcessId process = this->process(ppid);
    if ((process == this->photoelectric_process_id()
         && energy < params_.hardwired.photoelectric_table_thresh)
        || (process == this->eplusgg_process_id()))
    {
        auto find_model = this->make_model_finder(ppid);
        return this->model_id(find_model(energy));
    }
    // Not a hardwired process
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Particle-process ID of the process with the de/dx and range tables.
 */
CELER_FUNCTION ParticleProcessId PhysicsTrackView::eloss_ppid() const
{
    return this->process_group().eloss_ppid;
}

//---------------------------------------------------------------------------//
/*!
 * Particle-process ID of the multiple scattering process
 */
CELER_FUNCTION ParticleProcessId PhysicsTrackView::msc_ppid() const
{
    return this->process_group().msc_ppid;
}

//---------------------------------------------------------------------------//
/*!
 * Models that apply to the given process ID.
 */
CELER_FUNCTION auto
PhysicsTrackView::make_model_finder(ParticleProcessId ppid) const
    -> ModelFinder
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    const ModelGroup& md
        = params_.model_groups[this->process_group().models[ppid.get()]];
    return ModelFinder(params_.reals[md.energy], params_.pmodel_ids[md.model]);
}

//---------------------------------------------------------------------------//
/*!
 * Return value table data for the given particle/model/material.
 *
 * A null result means either the model is material independent or the material
 * only has one element, so no cross section CDF tables are stored.
 */
CELER_FUNCTION
ValueTableId PhysicsTrackView::value_table(ParticleModelId pmid) const
{
    CELER_EXPECT(pmid);

    // Get the model xs table for the given particle/model
    CELER_ASSERT(pmid < params_.model_xs.size());
    const ModelXsTable& model_xs = params_.model_xs[pmid];
    if (!model_xs)
        return {}; // No tables stored for this model

    // Get the value table for the current material
    CELER_ASSERT(material_ < model_xs.material.size());
    const auto& table_id_ref = model_xs.material[material_.get()];
    if (!table_id_ref)
        return {}; // Only one element in this material

    CELER_ASSERT(table_id_ref < params_.value_table_ids.size());
    return params_.value_table_ids[table_id_ref];
}

//---------------------------------------------------------------------------//
/*!
 * Construct an element selector to sample an element from tabulated xs data.
 */
CELER_FUNCTION
TabulatedElementSelector
PhysicsTrackView::make_element_selector(ValueTableId table_id,
                                        Energy       energy) const
{
    CELER_EXPECT(table_id < params_.value_tables.size());
    const ValueTable& table = params_.value_tables[table_id];
    return TabulatedElementSelector{table,
                                    params_.value_grids,
                                    params_.value_grid_ids,
                                    params_.reals,
                                    energy};
}

//---------------------------------------------------------------------------//
/*!
 * Whether the particle can have a discrete interaction at rest.
 */
CELER_FUNCTION bool PhysicsTrackView::has_at_rest() const
{
    return this->process_group().has_at_rest;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an action to a model ID for diagnostics, false if not a model.
 */
CELER_FUNCTION ModelId PhysicsTrackView::action_to_model(ActionId action) const
{
    if (!action)
        return ModelId{};

    // Rely on unsigned rollover if action ID is less than the first model
    ModelId::size_type result = action.unchecked_get()
                                - params_.scalars.model_to_action;
    if (result >= params_.scalars.num_models)
        return ModelId{};

    return ModelId{result};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a selected model ID into a simulation action ID.
 */
CELER_FUNCTION ActionId PhysicsTrackView::model_to_action(ModelId model) const
{
    CELER_ASSERT(model < params_.scalars.num_models);
    return ActionId{model.unchecked_get() + params_.scalars.model_to_action};
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID corresponding to the given ParticleModelId.
 */
CELER_FUNCTION ModelId PhysicsTrackView::model_id(ParticleModelId pmid) const
{
    CELER_EXPECT(pmid < params_.model_ids.size());
    return params_.model_ids[pmid];
}

//---------------------------------------------------------------------------//
/*!
 * Calculate scaled step range.
 *
 * This is the updated step function given by Eq. 7.4 of Geant4 Physics
 * Reference Manual, Release 10.6: \f[
   s = \alpha r + \rho (1 - \alpha) (2 - \frac{\rho}{r})
 \f]
 * where alpha is \c scaling_fraction and rho is \c scaling_min_range .
 *
 * Below scaling_min_range, no step scaling is applied, but the step can still
 * be arbitrarily small.
 */
CELER_FUNCTION real_type PhysicsTrackView::range_to_step(real_type range) const
{
    CELER_ASSERT(range >= 0);
    const real_type rho = params_.scalars.scaling_min_range;
    if (range < rho)
        return range;

    const real_type alpha = params_.scalars.scaling_fraction;
    range = alpha * range + rho * (1 - alpha) * (2 - rho / range);
    CELER_ENSURE(range > 0);
    return range;
}

//---------------------------------------------------------------------------//
/*!
 * Access scalar properties (options, IDs).
 */
CELER_FORCEINLINE_FUNCTION const PhysicsParamsScalars&
PhysicsTrackView::scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
/*!
 * Energy loss fluctuation model parameters.
 */
CELER_FUNCTION auto PhysicsTrackView::fluctuation() const
    -> const FluctuationRef&
{
    return params_.fluctuation;
}

//---------------------------------------------------------------------------//
/*!
 * Urban multiple scattering data
 */
CELER_FUNCTION auto PhysicsTrackView::urban_data() const -> const UrbanMscRef&
{
    return params_.hardwired.urban_data;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate macroscopic cross section on the fly.
 */
CELER_FUNCTION real_type PhysicsTrackView::calc_xs_otf(
    ModelId model, const MaterialView& material, Energy energy) const
{
    real_type result = 0;
    if (model == params_.hardwired.livermore_pe)
    {
        auto calc_xs = LivermorePEMacroXsCalculator(
            params_.hardwired.livermore_pe_data, material);
        result = calc_xs(energy);
    }
    else if (model == params_.hardwired.eplusgg)
    {
        auto calc_xs = EPlusGGMacroXsCalculator(params_.hardwired.eplusgg_data,
                                                material);
        result       = calc_xs(energy);
    }

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Number of particle types.
 */
CELER_FUNCTION size_type PhysicsTrackView::num_particles() const
{
    return params_.process_groups.size();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a grid calculator of the given type.
 *
 * The calculator must take two arguments: a reference to XsGridRef, and a
 * reference to the Values data structure.
 */
template<class T>
CELER_FUNCTION T PhysicsTrackView::make_calculator(ValueGridId id) const
{
    CELER_EXPECT(id < params_.value_grids.size());
    return T{params_.value_grids[id], params_.reals};
}

//---------------------------------------------------------------------------//
/*!
 * Process ID for photoelectric effect.
 */
CELER_FUNCTION ProcessId PhysicsTrackView::photoelectric_process_id() const
{
    return params_.hardwired.photoelectric;
}

//---------------------------------------------------------------------------//
/*!
 * Process ID for positron annihilation.
 */
CELER_FUNCTION ProcessId PhysicsTrackView::eplusgg_process_id() const
{
    return params_.hardwired.positron_annihilation;
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION HELPER FUNCTIONS
//---------------------------------------------------------------------------//
//! Get the thread-local state (mutable)
CELER_FUNCTION PhysicsTrackState& PhysicsTrackView::state()
{
    return states_.state[thread_];
}

//! Get the thread-local state (const)
CELER_FUNCTION const PhysicsTrackState& PhysicsTrackView::state() const
{
    return states_.state[thread_];
}

//! Get the group of processes that apply to the particle
CELER_FUNCTION const ProcessGroup& PhysicsTrackView::process_group() const
{
    CELER_EXPECT(particle_ < params_.process_groups.size());
    return params_.process_groups[particle_];
}

//---------------------------------------------------------------------------//
} // namespace celeritas

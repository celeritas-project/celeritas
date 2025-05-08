//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/xs/EPlusGGMacroXsCalculator.hh"
#include "celeritas/em/xs/LivermorePEMicroXsCalculator.hh"
#include "celeritas/grid/GridIdFinder.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/neutron/xs/NeutronElasticMicroXsCalculator.hh"
#include "celeritas/random/TabulatedElementSelector.hh"

#include "MacroXsCalculator.hh"
#include "ParticleTrackView.hh"
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
    //! \name Type aliases
    using Initializer_t = PhysicsTrackInitializer;
    using PhysicsParamsRef = NativeCRef<PhysicsParamsData>;
    using PhysicsStateRef = NativeRef<PhysicsStateData>;
    using Energy = units::MevEnergy;
    using ModelFinder = GridIdFinder<Energy, ParticleModelId>;
    //!@}

  public:
    // Construct from params, states, and per-state IDs
    inline CELER_FUNCTION PhysicsTrackView(PhysicsParamsRef const& params,
                                           PhysicsStateRef const& states,
                                           ParticleTrackView const& particle,
                                           PhysMatId material,
                                           TrackSlotId tid);

    // Initialize the track view
    inline CELER_FUNCTION PhysicsTrackView& operator=(Initializer_t const&);

    // Set the remaining MFP to interaction
    inline CELER_FUNCTION void interaction_mfp(real_type);

    // Reset the remaining MFP to interaction
    inline CELER_FUNCTION void reset_interaction_mfp();

    // Set the energy loss range for the current material and particle energy
    inline CELER_FUNCTION void dedx_range(real_type);

    // Set the range properties for multiple scattering
    inline CELER_FUNCTION void msc_range(MscRange const&);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Whether the remaining MFP has been calculated
    CELER_FORCEINLINE_FUNCTION bool has_interaction_mfp() const;

    // Remaining MFP to interaction [1]
    CELER_FORCEINLINE_FUNCTION real_type interaction_mfp() const;

    // Energy loss range for the current material and particle energy
    CELER_FORCEINLINE_FUNCTION real_type dedx_range() const;

    // Range properties for multiple scattering
    CELER_FORCEINLINE_FUNCTION MscRange const& msc_range() const;

    // Current material identifier
    CELER_FORCEINLINE_FUNCTION PhysMatId material_id() const;

    //// PROCESSES (depend on particle type and possibly material) ////

    // Number of processes that apply to this track
    inline CELER_FUNCTION ParticleProcessId::size_type
    num_particle_processes() const;

    // Process ID for the given within-particle process index
    inline CELER_FUNCTION ProcessId process(ParticleProcessId) const;

    // Get macro xs table, null if not present for this particle/material
    inline CELER_FUNCTION ValueGridId macro_xs_grid(ParticleProcessId) const;

    // Get energy loss table, null if not present for this particle/material
    inline CELER_FUNCTION ValueGridId energy_loss_grid() const;

    // Get range table, null if not present for this particle/material
    inline CELER_FUNCTION ValueGridId range_grid() const;

    // Get inverse range table, null if not present
    inline CELER_FUNCTION ValueGridId inverse_range_grid() const;

    // Get data for processes that use the integral approach
    inline CELER_FUNCTION IntegralXsProcess const&
    integral_xs_process(ParticleProcessId ppid) const;

    // Calculate macroscopic cross section for the process
    inline CELER_FUNCTION real_type calc_xs(ParticleProcessId ppid,
                                            MaterialView const& material,
                                            Energy energy) const;

    // Estimate maximum macroscopic cross section for the process over the step
    inline CELER_FUNCTION real_type calc_max_xs(IntegralXsProcess const& process,
                                                ParticleProcessId ppid,
                                                MaterialView const& material,
                                                Energy energy) const;

    // Models that apply to the given process ID
    inline CELER_FUNCTION
        ModelFinder make_model_finder(ParticleProcessId) const;

    // Return value table data for the given particle/model/material
    inline CELER_FUNCTION ValueTableId value_table(ParticleModelId) const;

    // Construct an element selector
    inline CELER_FUNCTION
        TabulatedElementSelector make_element_selector(ValueTableId,
                                                       Energy) const;

    // ID of the particle's at-rest process
    inline CELER_FUNCTION ParticleProcessId at_rest_process() const;

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
    CELER_FORCEINLINE_FUNCTION PhysicsParamsScalars const& scalars() const;

    // Access particle-dependent scalar properties
    CELER_FORCEINLINE_FUNCTION ParticleScalars const& particle_scalars() const;

    // Number of particle types
    inline CELER_FUNCTION size_type num_particles() const;

    // Construct a grid calculator from a physics table
    template<class T>
    inline CELER_FUNCTION T make_calculator(ValueGridId) const;

    //// HACKS ////

    // Get hardwired model, null if not present
    inline CELER_FUNCTION ModelId hardwired_model(ParticleProcessId ppid,
                                                  Energy energy) const;

  private:
    PhysicsParamsRef const& params_;
    PhysicsStateRef const& states_;
    ParticleId const particle_;
    PhysMatId const material_;
    TrackSlotId const track_slot_;
    bool is_heavy_;

    //// IMPLEMENTATION HELPER FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState& state();
    CELER_FORCEINLINE_FUNCTION PhysicsTrackState const& state() const;
    CELER_FORCEINLINE_FUNCTION ProcessGroup const& process_group() const;
    inline CELER_FUNCTION ValueGridId value_grid(ValueTableId) const;
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
PhysicsTrackView::PhysicsTrackView(PhysicsParamsRef const& params,
                                   PhysicsStateRef const& states,
                                   ParticleTrackView const& particle,
                                   PhysMatId mid,
                                   TrackSlotId tid)
    : params_(params)
    , states_(states)
    , particle_(particle.particle_id())
    , material_(mid)
    , track_slot_(tid)
    , is_heavy_(particle.is_heavy())
{
    CELER_EXPECT(track_slot_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the track view.
 */
CELER_FUNCTION PhysicsTrackView&
PhysicsTrackView::operator=(Initializer_t const&)
{
    this->state().interaction_mfp = 0;
    this->state().msc_range = {};
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Set the distance to the next interaction, in mean free paths.
 *
 * This value will be decremented at each step.
 */
CELER_FUNCTION void PhysicsTrackView::interaction_mfp(real_type mfp)
{
    CELER_EXPECT(mfp > 0);
    this->state().interaction_mfp = mfp;
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
 * Set the energy loss range for the current material and particle energy.
 *
 * This value will be calculated once at the beginning of each step.
 */
CELER_FUNCTION void PhysicsTrackView::dedx_range(real_type range)
{
    CELER_EXPECT(range > 0);
    this->state().dedx_range = range;
}

//---------------------------------------------------------------------------//
/*!
 * Set the range properties for multiple scattering.
 *
 * These values will be calculated at the first step in every tracking volume.
 */
CELER_FUNCTION void PhysicsTrackView::msc_range(MscRange const& msc_range)
{
    this->state().msc_range = msc_range;
}

//---------------------------------------------------------------------------//
/*!
 * Current material identifier.
 */
CELER_FUNCTION PhysMatId PhysicsTrackView::material_id() const
{
    return material_;
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
 * Energy loss range.
 */
CELER_FUNCTION real_type PhysicsTrackView::dedx_range() const
{
    real_type range = this->state().dedx_range;
    CELER_ENSURE(range > 0);
    return range;
}

//---------------------------------------------------------------------------//
/*!
 * Persistent range properties for multiple scattering within a same volume.
 */
CELER_FUNCTION MscRange const& PhysicsTrackView::msc_range() const
{
    return this->state().msc_range;
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
 * Return macro xs value grid data for the given process if available.
 */
CELER_FUNCTION auto
PhysicsTrackView::macro_xs_grid(ParticleProcessId ppid) const -> ValueGridId
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    return this->value_grid(this->process_group().macro_xs[ppid.get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Return the energy loss grid data if available.
 */
CELER_FUNCTION auto PhysicsTrackView::energy_loss_grid() const -> ValueGridId
{
    return this->value_grid(this->process_group().energy_loss);
}

//---------------------------------------------------------------------------//
/*!
 * Return the range grid data if available.
 */
CELER_FUNCTION auto PhysicsTrackView::range_grid() const -> ValueGridId
{
    return this->value_grid(this->process_group().range);
}

//---------------------------------------------------------------------------//
/*!
 * Return the inverse range grid data if available.
 *
 * If spline interpolation is used, the inverse grid is explicity stored with
 * the derivatives calculated using the range as the x values and the energy as
 * the y values.
 *
 * The grid and values are identical to the range grid (i.e., not inverted)
 * even if the inverse grid is explicitly stored: the inversion is done in the
 * \c InverseRangeCalculator .
 */
CELER_FUNCTION auto PhysicsTrackView::inverse_range_grid() const -> ValueGridId
{
    if (auto const& grid = this->value_grid(this->process_group().inverse_range))
    {
        return grid;
    }
    return this->range_grid();
}

//---------------------------------------------------------------------------//
/*!
 * Get data for processes that use the integral approach.
 *
 * Particles that have energy loss processes will have a different energy at
 * the pre- and post-step points. This means the assumption that the cross
 * section is constant along the step is no longer valid. Instead, Monte Carlo
 * integration can be used to sample the interaction length for the discrete
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
CELER_FUNCTION auto PhysicsTrackView::integral_xs_process(
    ParticleProcessId ppid) const -> IntegralXsProcess const&
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    return params_.integral_xs[this->process_group().integral_xs[ppid.get()]];
}

//---------------------------------------------------------------------------//
/*!
 * Calculate macroscopic cross section for the process.
 */
CELER_FUNCTION real_type PhysicsTrackView::calc_xs(ParticleProcessId ppid,
                                                   MaterialView const& material,
                                                   Energy energy) const
{
    real_type result = 0;

    if (auto model_id = this->hardwired_model(ppid, energy))
    {
        // Calculate macroscopic cross section on the fly for special
        // hardwired processes.
        if (model_id == params_.hardwired.livermore_pe)
        {
            auto calc_xs = MacroXsCalculator<LivermorePEMicroXsCalculator>(
                params_.hardwired.livermore_pe_data, material);
            result = calc_xs(energy);
        }
        else if (model_id == params_.hardwired.eplusgg)
        {
            auto calc_xs = EPlusGGMacroXsCalculator(
                params_.hardwired.eplusgg_data, material);
            result = calc_xs(energy);
        }
        else if (model_id == params_.hardwired.chips)
        {
            auto calc_xs = MacroXsCalculator<NeutronElasticMicroXsCalculator>(
                params_.hardwired.chips_data, material);
            result = calc_xs(energy);
        }
    }
    else if (auto grid_id = this->macro_xs_grid(ppid))
    {
        // Calculate cross section from the tabulated data
        auto calc_xs = this->make_calculator<XsCalculator>(grid_id);
        result = calc_xs(energy);
    }

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Estimate maximum macroscopic cross section for the process over the step.
 *
 * If this is a particle with an energy loss process, this returns the
 * estimate of the maximum cross section over the step. If the energy of the
 * global maximum of the cross section (calculated at initialization) is in the
 * interval \f$ [\xi E_0, E_0) \f$, where \f$ E_0 \f$ is the pre-step energy
 * and \f$ \xi \f$ is \c min_eprime_over_e (defined by default as \f$ \xi = 1 -
 * \alpha \f$, where \f$ \alpha \f$ is \c max_step_over_range), \f$
 * \sigma_{\max} \f$ is set to the global maximum.  Otherwise, \f$
 * \sigma_{\max} = \max( \sigma(E_0), \sigma(\xi E_0) ) \f$. If the cross
 * section is not monotonic in the interval \f$ [\xi E_0, E_0) \f$ and the
 * interval does not contain the global maximum, the post-step cross section
 * \f$ \sigma(E_1) \f$ may be larger than \f$ \sigma_{\max} \f$.
 */
CELER_FUNCTION real_type
PhysicsTrackView::calc_max_xs(IntegralXsProcess const& process,
                              ParticleProcessId ppid,
                              MaterialView const& material,
                              Energy energy) const
{
    CELER_EXPECT(process);
    CELER_EXPECT(material_ < process.energy_max_xs.size());

    real_type energy_max_xs
        = params_.reals[process.energy_max_xs[material_.get()]];
    real_type energy_xi = energy.value() * params_.scalars.min_eprime_over_e;
    if (energy_max_xs >= energy_xi && energy_max_xs < energy.value())
    {
        return this->calc_xs(ppid, material, Energy{energy_max_xs});
    }
    return max(this->calc_xs(ppid, material, energy),
               this->calc_xs(ppid, material, Energy{energy_xi}));
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
    if ((process == params_.hardwired.photoelectric
         && energy < params_.hardwired.photoelectric_table_thresh)
        || (process == params_.hardwired.positron_annihilation)
        || (process == params_.hardwired.neutron_elastic))
    {
        auto find_model = this->make_model_finder(ppid);
        return this->model_id(find_model(energy));
    }
    // Not a hardwired process
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Models that apply to the given process ID.
 */
CELER_FUNCTION auto
PhysicsTrackView::make_model_finder(ParticleProcessId ppid) const -> ModelFinder
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    ModelGroup const& md
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
    CELER_EXPECT(pmid < params_.model_cdf.size());

    // Get the model xs table for the given particle/model
    ModelCdfTable const& model_cdf = params_.model_cdf[pmid];
    if (!model_cdf)
        return {};  // No tables stored for this model

    // Get the value table for the current material
    CELER_ASSERT(material_ < model_cdf.material.size());
    auto const& table_id_ref = model_cdf.material[material_.get()];
    if (!table_id_ref)
        return {};  // Only one element in this material

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
                                        Energy energy) const
{
    CELER_EXPECT(table_id < params_.value_tables.size());
    ValueTable const& table = params_.value_tables[table_id];
    return TabulatedElementSelector{table,
                                    params_.value_grids,
                                    params_.value_grid_ids,
                                    params_.reals,
                                    energy};
}

//---------------------------------------------------------------------------//
/*!
 * ID of the particle's at-rest process.
 *
 * If the partcle can have a discrete interaction at rest, this returns the \c
 * ParticleProcessId of that process. Otherwise, it returns an invalid ID.
 */
CELER_FUNCTION ParticleProcessId PhysicsTrackView::at_rest_process() const
{
    return this->process_group().at_rest;
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
 * where alpha is \c max_step_over_range and rho is \c min_range .
 *
 * Below \c min_range, no step scaling is applied, but the step can still
 * be arbitrarily small.
 */
CELER_FUNCTION real_type PhysicsTrackView::range_to_step(real_type range) const
{
    CELER_ASSERT(range >= 0);
    auto const& scalars = this->particle_scalars();
    real_type const rho = scalars.min_range;
    if (range < rho * (1 + celeritas::sqrt_tol()))
    {
        // Small range returns the step. The fudge factor avoids floating point
        // error in the interpolation below while preserving the near-linear
        // behavior for range = rho + epsilon.
        return range;
    }

    real_type const alpha = scalars.max_step_over_range;
    real_type step = alpha * range + rho * (1 - alpha) * (2 - rho / range);
    CELER_ENSURE(step > 0 && step <= range);
    return step;
}

//---------------------------------------------------------------------------//
/*!
 * Access scalar properties (options, IDs).
 */
CELER_FORCEINLINE_FUNCTION PhysicsParamsScalars const&
PhysicsTrackView::scalars() const
{
    return params_.scalars;
}

//---------------------------------------------------------------------------//
/*!
 * Access particle-dependent scalar properties.
 *
 * These properties are different for light particles (electrons/positrons) and
 * heavy particles (muons/hadrons).
 */
CELER_FORCEINLINE_FUNCTION ParticleScalars const&
PhysicsTrackView::particle_scalars() const
{
    if (is_heavy_)
    {
        return params_.scalars.heavy;
    }
    return params_.scalars.light;
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
// IMPLEMENTATION HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return value grid data for the given table ID if available.
 *
 * If the result is not null, it can be used to instantiate a
 * grid Calculator.
 *
 * If the result is null, it's likely because the process doesn't have the
 * associated value (e.g. if the table type is "energy_loss" and the process is
 * not a slowing-down process).
 */
CELER_FUNCTION auto PhysicsTrackView::value_grid(ValueTableId table_id) const
    -> ValueGridId
{
    CELER_EXPECT(table_id);

    ValueTable const& table = params_.value_tables[table_id];
    if (!table)
        return {};  // No table for this process

    CELER_EXPECT(material_ < table.grids.size());
    auto grid_id_ref = table.grids[material_.get()];
    if (!grid_id_ref)
        return {};  // No table for this particular material

    return params_.value_grid_ids[grid_id_ref];
}

//! Get the thread-local state (mutable)
CELER_FUNCTION PhysicsTrackState& PhysicsTrackView::state()
{
    return states_.state[track_slot_];
}

//! Get the thread-local state (const)
CELER_FUNCTION PhysicsTrackState const& PhysicsTrackView::state() const
{
    return states_.state[track_slot_];
}

//! Get the group of processes that apply to the particle
CELER_FUNCTION ProcessGroup const& PhysicsTrackView::process_group() const
{
    CELER_EXPECT(particle_ < params_.process_groups.size());
    return params_.process_groups[particle_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

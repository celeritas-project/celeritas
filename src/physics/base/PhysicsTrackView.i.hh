//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTrackView.i.hh
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "physics/em/EPlusGGMacroXsCalculator.hh"
#include "physics/em/LivermorePEMacroXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and static data.
 */
CELER_FUNCTION
PhysicsTrackView::PhysicsTrackView(const PhysicsParamsPointers& params,
                                   const PhysicsStatePointers&  states,
                                   ParticleId                   pid,
                                   MaterialId                   mid,
                                   ThreadId                     tid)
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
 *
 * \todo Add total interaction cross section to state.
 */
CELER_FUNCTION PhysicsTrackView&
PhysicsTrackView::operator=(const Initializer_t&)
{
    this->state().interaction_mfp = -1;
    this->state().step_length     = -1;
    this->state().macro_xs        = -1;
    this->state().model_id        = ModelId{};
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
 * Set the remaining physics step length.
 */
CELER_FUNCTION void PhysicsTrackView::step_length(real_type distance)
{
    CELER_EXPECT(distance > 0);
    this->state().step_length = distance;
}

//---------------------------------------------------------------------------//
/*!
 * Set the process-integrated total macroscopic cross section.
 */
CELER_FUNCTION void PhysicsTrackView::macro_xs(real_type inv_distance)
{
    CELER_EXPECT(inv_distance > 0);
    this->state().macro_xs = inv_distance;
}

//---------------------------------------------------------------------------//
/*!
 * Select a model ID for the current track.
 *
 * An "unassigned" model ID is valid, as it might represent a special case or a
 * particle that is not undergoing an interaction.
 */
CELER_FUNCTION void PhysicsTrackView::model_id(ModelId id)
{
    this->state().model_id = id;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the remaining MFP has been calculated.
 */
CELER_FUNCTION bool PhysicsTrackView::has_interaction_mfp() const
{
    return this->state().interaction_mfp >= 0;
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
 * Maximum step length.
 */
CELER_FUNCTION real_type PhysicsTrackView::step_length() const
{
    real_type length = this->state().step_length;
    CELER_ENSURE(length >= 0);
    return length;
}

//---------------------------------------------------------------------------//
/*!
 * Calculated process-integrated macroscopic XS.
 *
 * This value should be calculated in the pre-step kernel, and will be used to
 * decrement `interaction_mfp` and for sampling a process.
 */
CELER_FUNCTION real_type PhysicsTrackView::macro_xs() const
{
    real_type xs = this->state().macro_xs;
    CELER_ENSURE(xs >= 0);
    return xs;
}

//---------------------------------------------------------------------------//
/*!
 * Access the model ID that has been selected for the current track.
 *
 * If no model applies (e.g. if the particle has exited the geometry) the
 * result will be the \c ModelId() which evaluates to false.
 */
CELER_FUNCTION ModelId PhysicsTrackView::model_id() const
{
    return this->state().model_id;
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
        = this->process_group().tables[int(table_type)][ppid.get()];

    CELER_ASSERT(table_id);
    const ValueTable& table = params_.value_tables[table_id];
    if (!table)
        return {}; // No table for this process

    CELER_EXPECT(material_ < table.material.size());
    auto grid_id_ref = table.material[material_.get()];
    if (!grid_id_ref)
        return {}; // No table for this particular material

    return params_.value_grid_ids[grid_id_ref];
}

//---------------------------------------------------------------------------//
/*!
 * Return the model ID that applies to the given process ID and energy if the
 * process is hardwired to calculate macroscopic cross sections on the fly. If
 * the result is null, tables should be used for this process/energy.
 */
CELER_FUNCTION ModelId PhysicsTrackView::hardwired_model(ParticleProcessId ppid,
                                                         MevEnergy energy) const
{
    ProcessId process = this->process(ppid);
    if ((process == this->photoelectric_process_id()
         && energy < params_.hardwired.photoelectric_table_thresh)
        || (process == this->eplusgg_process_id()))
    {
        auto find_model = this->make_model_finder(ppid);
        return find_model(energy);
    }
    // Not a hardwired process
    return {};
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
    const ModelGroup& mg
        = params_.model_groups[this->process_group().models[ppid.get()]];
    return ModelFinder(params_.reals[mg.energy], params_.model_ids[mg.model]);
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
    CELER_EXPECT(range > 0);

    const real_type rho = params_.scaling_min_range;
    if (range < rho)
        return range;

    const real_type alpha = params_.scaling_fraction;
    range = alpha * range + rho * (1 - alpha) * (2 - rho / range);
    CELER_ENSURE(range > 0);
    return range;
}

//---------------------------------------------------------------------------//
/*!
 * Fractional along-step energy loss allowed before recalculating from range.
 */
CELER_FUNCTION real_type PhysicsTrackView::linear_loss_limit() const
{
    return params_.linear_loss_limit;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate macroscopic cross section on the fly.
 */
CELER_FUNCTION real_type PhysicsTrackView::calc_xs_otf(ModelId       model,
                                                       MaterialView& material,
                                                       MevEnergy energy) const
{
    real_type result = 0.;
    if (model == params_.hardwired.livermore_pe)
    {
        auto calc_xs = LivermorePEMacroXsCalculator(
            params_.hardwired.livermore_pe_params, material);
        result = calc_xs(energy);
    }
    else if (model == params_.hardwired.eplusgg)
    {
        auto calc_xs = EPlusGGMacroXsCalculator(
            params_.hardwired.eplusgg_params, material);
        result = calc_xs(energy);
    }

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a grid calculator of the given type.
 *
 * The calculator must take two arguments: a reference to XsGridData, and a
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
 * Access scratch space for particle-process cross section calculations.
 */
CELER_FUNCTION real_type&
               PhysicsTrackView::per_process_xs(ParticleProcessId ppid)
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    auto idx = thread_.get() * params_.max_particle_processes + ppid.get();
    CELER_ENSURE(idx < states_.per_process_xs.size());
    return states_.per_process_xs[ItemId<real_type>(idx)];
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space for particle-process cross section calculations.
 */
CELER_FUNCTION
real_type PhysicsTrackView::per_process_xs(ParticleProcessId ppid) const
{
    CELER_EXPECT(ppid < this->num_particle_processes());
    auto idx = thread_.get() * params_.max_particle_processes + ppid.get();
    CELER_ENSURE(idx < states_.per_process_xs.size());
    return states_.per_process_xs[ItemId<real_type>(idx)];
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

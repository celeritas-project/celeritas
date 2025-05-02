//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/em/interactor/AtomicRelaxationHelper.hh"

#include "Interaction.hh"
#include "PhysicsData.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access step-local (non-persistent) physics track data.
 *
 * This class should be accessible to track slots that have inactive tracks so
 * that the temporary values can be cleared. (Once we start partitioning track
 * slots based on their status, this restriction might be lifted.) Unlike
 * \c PhysicsTrackView, this class shouldn't need any external info about the
 * current particle type, material, etc.
 *
 * The underlying physics data might be refactored later (and this class name
 * might be changed) but it separates out some of the temporary data
 * (interaction, macro xs, MSC step) from the persistent data (remaining MFP).
 */
class PhysicsStepView
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsParamsRef = NativeCRef<PhysicsParamsData>;
    using PhysicsStateRef = NativeRef<PhysicsStateData>;
    using SecondaryAllocator = StackAllocator<Secondary>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from shared and state data
    inline CELER_FUNCTION PhysicsStepView(PhysicsParamsRef const& params,
                                          PhysicsStateRef const& states,
                                          TrackSlotId tid);

    // Set the total (process-integrated) macroscopic xs [len^-1]
    inline CELER_FUNCTION void macro_xs(real_type);

    // Set the sampled element
    inline CELER_FUNCTION void element(ElementComponentId);

    // Save MSC step data
    inline CELER_FUNCTION void msc_step(MscStep const&);

    // Reset the energy deposition
    inline CELER_FUNCTION void reset_energy_deposition();

    // Reset the energy deposition to NaN to catch errors
    inline CELER_FUNCTION void reset_energy_deposition_debug();

    // Accumulate into local step's energy deposition
    inline CELER_FUNCTION void deposit_energy(Energy);

    // Set secondaries during an interaction
    inline CELER_FUNCTION void secondaries(Span<Secondary>);

    // Total (process-integrated) macroscopic xs [len^-1]
    CELER_FORCEINLINE_FUNCTION real_type macro_xs() const;

    // Sampled element for discrete interaction
    CELER_FORCEINLINE_FUNCTION ElementComponentId element() const;

    // Mutable access to MSC step data (TODO: hack)
    inline CELER_FUNCTION MscStep& msc_step();

    // Retrieve MSC step data
    inline CELER_FUNCTION MscStep const& msc_step() const;

    // Access local energy deposition
    inline CELER_FUNCTION Energy energy_deposition() const;

    // Access secondaries created by an interaction
    inline CELER_FUNCTION Span<Secondary const> secondaries() const;

    // Access scratch space for particle-process cross section calculations
    inline CELER_FUNCTION real_type& per_process_xs(ParticleProcessId);
    inline CELER_FUNCTION real_type per_process_xs(ParticleProcessId) const;

    //// THREAD-INDEPENDENT ////

    // Return a secondary stack allocator
    inline CELER_FUNCTION SecondaryAllocator make_secondary_allocator() const;

    // Access atomic relaxation data
    inline CELER_FUNCTION AtomicRelaxationHelper
    make_relaxation_helper(ElementId el_id) const;

  private:
    //// DATA ////

    PhysicsParamsRef const& params_;
    PhysicsStateRef const& states_;
    TrackSlotId const track_slot_;

    //// CLASS FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState& state();
    CELER_FORCEINLINE_FUNCTION PhysicsTrackState const& state() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and state data.
 */
CELER_FUNCTION PhysicsStepView::PhysicsStepView(PhysicsParamsRef const& params,
                                                PhysicsStateRef const& states,
                                                TrackSlotId tid)
    : params_(params), states_(states), track_slot_(tid)
{
    CELER_EXPECT(track_slot_);
}

//---------------------------------------------------------------------------//
/*!
 * Set the process-integrated total macroscopic cross section.
 */
CELER_FUNCTION void PhysicsStepView::macro_xs(real_type inv_distance)
{
    CELER_EXPECT(inv_distance >= 0);
    this->state().macro_xs = inv_distance;
}

//---------------------------------------------------------------------------//
/*!
 * Set the sampled element.
 */
CELER_FUNCTION void PhysicsStepView::element(ElementComponentId elcomp_id)
{
    this->state().element = elcomp_id;
}

//---------------------------------------------------------------------------//
/*!
 * Save MSC step limit data.
 */
CELER_FUNCTION void PhysicsStepView::msc_step(MscStep const& limit)
{
    states_.msc_step[track_slot_] = limit;
}

//---------------------------------------------------------------------------//
/*!
 * Reset the energy deposition to zero at the beginning of a step.
 */
CELER_FUNCTION void PhysicsStepView::reset_energy_deposition()
{
    this->state().energy_deposition = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Set the energy deposition to NaN for inactive tracks to catch errors.
 */
CELER_FUNCTION void PhysicsStepView::reset_energy_deposition_debug()
{
#if !CELERITAS_DEBUG
    CELER_ASSERT_UNREACHABLE();
#endif
    this->state().energy_deposition = numeric_limits<real_type>::quiet_NaN();
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate into local step's energy deposition.
 */
CELER_FUNCTION void PhysicsStepView::deposit_energy(Energy energy)
{
    CELER_EXPECT(energy >= zero_quantity());
    // TODO: save a memory read/write by skipping if energy is zero?
    this->state().energy_deposition += energy.value();
}

//---------------------------------------------------------------------------//
/*!
 * Set secondaries during an interaction, or clear them with an empty span.
 */
CELER_FUNCTION void PhysicsStepView::secondaries(Span<Secondary> sec)
{
    this->state().secondaries = sec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculated process-integrated macroscopic XS.
 *
 * This value should be calculated in the pre-step kernel, and will be used to
 * decrement `interaction_mfp` and for sampling a process. Units are inverse
 * length.
 */
CELER_FUNCTION real_type PhysicsStepView::macro_xs() const
{
    real_type xs = this->state().macro_xs;
    CELER_ENSURE(xs >= 0);
    return xs;
}

//---------------------------------------------------------------------------//
/*!
 * Sampled element for discrete interaction.
 */
CELER_FUNCTION ElementComponentId PhysicsStepView::element() const
{
    return this->state().element;
}

//---------------------------------------------------------------------------//
/*!
 * Mutable access to MSC step data (TODO: hack)
 */
CELER_FUNCTION MscStep& PhysicsStepView::msc_step()
{
    return states_.msc_step[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Access calculated MSC step data.
 */
CELER_FUNCTION MscStep const& PhysicsStepView::msc_step() const
{
    return states_.msc_step[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Access accumulated energy deposition.
 */
CELER_FUNCTION auto PhysicsStepView::energy_deposition() const -> Energy
{
    real_type result = this->state().energy_deposition;
    CELER_ENSURE(result >= 0);
    return Energy{result};
}

//---------------------------------------------------------------------------//
/*!
 * Access secondaries created by a discrete interaction.
 */
CELER_FUNCTION Span<Secondary const> PhysicsStepView::secondaries() const
{
    return this->state().secondaries;
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space for particle-process cross section calculations.
 */
CELER_FUNCTION real_type&
PhysicsStepView::per_process_xs(ParticleProcessId ppid)
{
    CELER_EXPECT(ppid < params_.scalars.max_particle_processes);
    auto idx = track_slot_.get() * params_.scalars.max_particle_processes
               + ppid.get();
    CELER_ENSURE(idx < states_.per_process_xs.size());
    return states_.per_process_xs[ItemId<real_type>(idx)];
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space for particle-process cross section calculations.
 */
CELER_FUNCTION
real_type PhysicsStepView::per_process_xs(ParticleProcessId ppid) const
{
    CELER_EXPECT(ppid < params_.scalars.max_particle_processes);
    auto idx = track_slot_.get() * params_.scalars.max_particle_processes
               + ppid.get();
    CELER_ENSURE(idx < states_.per_process_xs.size());
    return states_.per_process_xs[ItemId<real_type>(idx)];
}

//---------------------------------------------------------------------------//
/*!
 * Return a secondary stack allocator view.
 */
CELER_FUNCTION auto PhysicsStepView::make_secondary_allocator() const
    -> SecondaryAllocator
{
    return SecondaryAllocator{states_.secondaries};
}

//---------------------------------------------------------------------------//
/*!
 * Make an atomic relaxation helper for the given element.
 */
CELER_FUNCTION auto
PhysicsStepView::make_relaxation_helper(ElementId el_id) const
    -> AtomicRelaxationHelper
{
    CELER_ASSERT(el_id);
    return AtomicRelaxationHelper{params_.hardwired.relaxation_data,
                                  states_.relaxation,
                                  el_id,
                                  track_slot_};
}

//---------------------------------------------------------------------------//
// PRIVATE CLASS FUNCTIONS
//---------------------------------------------------------------------------//
//! Get the thread-local state (mutable)
CELER_FUNCTION PhysicsTrackState& PhysicsStepView::state()
{
    return states_.state[track_slot_];
}

//! Get the thread-local state (const)
CELER_FUNCTION PhysicsTrackState const& PhysicsStepView::state() const
{
    return states_.state[track_slot_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

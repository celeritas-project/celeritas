//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "PhysicsData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Access step-local (non-persistent) physics track data.
 */
class PhysicsStepView
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsParamsRef = NativeCRef<PhysicsParamsData>;
    using PhysicsStateRef = NativeRef<PhysicsStateData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from shared and state data
    inline CELER_FUNCTION PhysicsStepView(PhysicsParamsRef const& params,
                                          PhysicsStateRef const& states,
                                          TrackSlotId tid);

    //// MACRO XS ////

    // Set the total (process-integrated) macroscopic xs [len^-1]
    inline CELER_FUNCTION void macro_xs(real_type);

    // Total macroscopic xs [len^-1]
    inline CELER_FUNCTION real_type macro_xs() const;

    // // Access per model scratch space for calculating cross sections
    // inline CELER_FUNCTION real_type& per_model_xs(ModelId);
    // inline CELER_FUNCTION real_type per_model_xs(ModelId) const;

    //// ENERGY DEPOSITION ////

    // Reset the energy deposition
    inline CELER_FUNCTION void reset_energy_deposition();

    // Accumulate into local step's energy deposition
    inline CELER_FUNCTION void deposit_energy(Energy);

    // Local energy deposition
    inline CELER_FUNCTION Energy energy_deposition() const;

  private:
    PhysicsParamsRef const& params_;
    PhysicsStateRef const& states_;
    TrackSlotId const track_slot_;

    //// IMPLEMENTATION HELPER FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState& state();
    CELER_FORCEINLINE_FUNCTION PhysicsTrackState const& state() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct the step view from shared and state data.
 */
CELER_FUNCTION PhysicsStepView::PhysicsStepView(PhysicsParamsRef const& params,
                                                PhysicsStateRef const& states,
                                                TrackSlotId tid)
    : params_(params), states_(states), track_slot_(tid)
{
    CELER_EXPECT(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Set the total (process-integrated) macroscopic cross section [len^-1].
 */
CELER_FUNCTION void PhysicsStepView::macro_xs(real_type xs)
{
    CELER_ASSERT(xs >= 0);
    this->state().macro_xs = xs;
}

//---------------------------------------------------------------------------//
/*!
 * Get the total macroscopic cross section [len^-1].
 */
CELER_FUNCTION real_type PhysicsStepView::macro_xs() const
{
    return this->state().macro_xs;
}

// //---------------------------------------------------------------------------//
// /*!
//  * Access scratch space used for per-model cross section calculations.
//  */
// CELER_FUNCTION real_type& PhysicsStepView::per_model_xs(ModelId)
// {
// }
//
// //---------------------------------------------------------------------------//
// /*!
//  */
// CELER_FUNCTION real_type PhysicsStepView::per_model_xs(ModelId) const;

//---------------------------------------------------------------------------//
/*!
 * Resets the energy deposition of the step.
 */
CELER_FUNCTION void PhysicsStepView::reset_energy_deposition()
{
    this->state().energy_deposition = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate the energy into the local step's total energy deposition.
 */
CELER_FUNCTION void PhysicsStepView::deposit_energy(Energy energy)
{
    this->state().energy_deposition += value_as<Energy>(energy);
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the total local energy deposition in the step.
 */
CELER_FUNCTION auto PhysicsStepView::energy_deposition() const -> Energy
{
    return Energy{this->state().energy_deposition};
}

//---------------------------------------------------------------------------//

//! Access the state associated with the track
CELER_FUNCTION PhysicsTrackState& PhysicsStepView::state()
{
    return states_.states[track_slot_];
}

//! Access the state associated with the track
CELER_FUNCTION PhysicsTrackState const& PhysicsStepView::state() const
{
    return states_.states[track_slot_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

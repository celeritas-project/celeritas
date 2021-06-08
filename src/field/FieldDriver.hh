//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldDriver.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

#include "RungeKuttaStepper.hh"
#include "MagFieldEquation.hh"
#include "FieldParamsPointers.hh"
#include "FieldInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Integrate with and control the quality of the field integration stepper.
 *
 * \note This class is based on G4ChordFinder and G4MagIntegratorDriver.
 */
class FieldDriver
{
  public:
    // Construct with shared data and the stepper
    inline CELER_FUNCTION
    FieldDriver(const FieldParamsPointers&           shared,
                RungeKuttaStepper<MagFieldEquation>& stepper);

    // For a given trial step, advance by a sub_step within a tolerance error
    inline CELER_FUNCTION real_type operator()(real_type step, OdeState* state);

    // An adaptive step size control from G4MagIntegratorDriver
    // Move this to private after all tests with non-uniform field are done
    inline CELER_FUNCTION real_type accurate_advance(real_type step,
                                                     OdeState* state,
                                                     real_type hinitial);

    //// AUXILIARY INTERFACE ////

    CELER_FUNCTION real_type minimum_step() const
    {
        return shared_.minimum_step;
    }

    CELER_FUNCTION real_type max_nsteps() const { return shared_.max_nsteps; }

    CELER_FUNCTION real_type delta_intersection() const
    {
        return shared_.delta_intersection;
    }

  private:
    //// DATA ////

    // Shared constant properties
    const FieldParamsPointers& shared_;
    // Stepper for this field driver
    RungeKuttaStepper<MagFieldEquation>& stepper_;

    //// CONSTANTS ////

    static CELER_CONSTEXPR_FUNCTION real_type half() { return 0.5; }

    static CELER_CONSTEXPR_FUNCTION real_type ppm() { return 1e-6; }

    //// HELPER TYPES ////

    // A helper output for private member functions
    struct FieldOutput
    {
        real_type step_taken; //!< Step length taken
        OdeState  state;      //!< OdeState
        union
        {
            real_type error;     //!< Stepper error
            real_type next_step; //!< Proposed next step size
        };
    };

    //// HELPER FUNCTIONS ////

    // Find the next acceptable chord of with the miss-distance
    inline CELER_FUNCTION auto
    find_next_chord(real_type step, const OdeState& state) -> FieldOutput;

    // Advance for a given step and  evaluate the next predicted step.
    inline CELER_FUNCTION auto
    integrate_step(real_type step, const OdeState& state) -> FieldOutput;

    // Advance within the truncated error and estimate a good next step size
    inline CELER_FUNCTION auto
    one_good_step(real_type step, const OdeState& state) -> FieldOutput;

    // Propose a next step size from a given step size and associated error
    inline CELER_FUNCTION real_type new_step_size(real_type step,
                                                  real_type error) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldDriver.i.hh"

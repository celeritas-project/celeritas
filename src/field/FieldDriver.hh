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
 * Brief class description.
 *
 *  This is a driver to control the quality of the field integration stepper
 *  and provides the integration with a given field stepper
 *
 * \note This class is based on G4ChordFinder and G4MagIntegratorDriver
 */
class FieldDriver
{
  public:
    // Construct with shared data and the stepper
    inline CELER_FUNCTION
    FieldDriver(const FieldParamsPointers&           shared,
                RungeKuttaStepper<MagFieldEquation>& stepper);

    // For a given trial step, advance by a sub_step within a tolerance error
    CELER_FUNCTION real_type operator()(real_type step, OdeState* state);

    // An adaptive step size control from G4MagIntegratorDriver
    // Move this to private after all tests with non-uniform field are done
    CELER_FUNCTION real_type accurate_advance(real_type step,
                                              OdeState* state,
                                              real_type hinitial);

  private:
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

    // Find the next acceptable chord of with the miss-distance
    CELER_FUNCTION auto find_next_chord(real_type step, const OdeState& state)
        -> FieldOutput;

    // Advance for a given step and  evaluate the next predicted step.
    CELER_FUNCTION auto integrate_step(real_type step, const OdeState& state)
        -> FieldOutput;

    // Advance within the truncated error and estimate a good next step size
    CELER_FUNCTION auto one_good_step(real_type step, const OdeState& state)
        -> FieldOutput;

    // Propose a next step size from a given step size and associated error
    CELER_FUNCTION real_type new_step_size(real_type step,
                                           real_type error) const;

    // >>> COMMON PROPERTIES

    static CELER_CONSTEXPR_FUNCTION real_type half() { return 0.5; }

    static CELER_CONSTEXPR_FUNCTION real_type rel_tolerance() { return 1e-6; }

  private:
    // Shared constant properties
    const FieldParamsPointers& shared_;

    // Stepper for this field driver
    RungeKuttaStepper<MagFieldEquation>& stepper_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldDriver.i.hh"

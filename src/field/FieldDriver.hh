//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
#include "FieldParamsData.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Integrate with and control the quality of the field integration stepper.
 *
 * \note This class is based on G4ChordFinder and G4MagIntegratorDriver.
 */
template<class StepperT>
class FieldDriver
{
  public:
    // Construct with shared data and the stepper
    inline CELER_FUNCTION
    FieldDriver(const FieldParamsData& shared, StepperT* stepper);

    // For a given trial step, advance by a sub_step within a tolerance error
    inline CELER_FUNCTION DriverResult advance(real_type       step,
                                               const OdeState& state);

    // An adaptive step size control from G4MagIntegratorDriver
    // Move this to private after all tests with non-uniform field are done
    inline CELER_FUNCTION DriverResult accurate_advance(real_type       step,
                                                        const OdeState& state,
                                                        real_type hinitial);

    //// ACCESSORS ////

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
    const FieldParamsData& shared_;

    // Stepper for this field driver
    StepperT& stepper_;

    //// TYPES ////

    //! A helper output for private member functions
    struct ChordSearch
    {
        DriverResult end;   //!< Step taken and post-step state
        real_type    error; //!< Stepper error
    };

    struct Integration
    {
        DriverResult end;           //!< Step taken and post-step state
        real_type    proposed_step; //!< Proposed next step size
    };

    //// HEPER FUNCTIONS ////

    // Find the next acceptable chord of with the miss-distance
    inline CELER_FUNCTION ChordSearch find_next_chord(real_type       step,
                                                      const OdeState& state);

    // Advance for a given step and  evaluate the next predicted step.
    inline CELER_FUNCTION Integration integrate_step(real_type       step,
                                                     const OdeState& state);

    // Advance within the truncated error and estimate a good next step size
    inline CELER_FUNCTION Integration one_good_step(real_type       step,
                                                    const OdeState& state);

    // Propose a next step size from a given step size and associated error
    inline CELER_FUNCTION real_type new_step_size(real_type step,
                                                  real_type error) const;

    //// COMMON PROPERTIES ////

    static CELER_CONSTEXPR_FUNCTION real_type half() { return 0.5; }

    static CELER_CONSTEXPR_FUNCTION real_type ppm() { return 1e-6; }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldDriver.i.hh"

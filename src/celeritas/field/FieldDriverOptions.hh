//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriverOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Configuration options for the field driver.
 *
 * TODO: replace epsilon_rel_max with 1/epsilon_rel_max^2
 * TODO: replace safety with step_shrink_mul (or something to indicate that
 *       it's a multiplicative factor for reducing the step, not anything with
 *       geometry)
 * TODO: remove errcon
 * TODO: for some of these we could probably use single-precision
 */
struct FieldDriverOptions
{
    //! The minimum length of the field step
    real_type minimum_step = 1.0e-5 * units::millimeter;

    //! The maximum sagitta of each substep ("miss distance")
    real_type delta_chord = 0.25 * units::millimeter;

    //! Accuracy of intersection of the boundary crossing
    real_type delta_intersection = 1.0e-4 * units::millimeter;

    //! Discretization error tolerance for each field substep
    real_type epsilon_step = 1.0e-5;

    //! Targeted discretization error for "integrate step"
    real_type epsilon_rel_max = 1.0e-3;

    //! UNUSED: Targeted discretization error for "one good step"
    real_type errcon = 1.0e-4;

    //! Exponent to increase a step size
    real_type pgrow = -0.20;

    //! Exponent to decrease a step size
    real_type pshrink = -0.25;

    //! Scale factor for the predicted step size
    real_type safety = 0.9;

    //! Largest allowable relative increase a step size
    real_type max_stepping_increase = 5;

    //! Smallest allowable relative decrease in step size
    real_type max_stepping_decrease = 0.1;

    //! Maximum number of integrations (or trials)
    short int max_nsteps = 100;

    //! Maximum number of substeps in the field propagator
    short int max_substeps = 10;

    //! Initial step tolerance
    static constexpr real_type initial_step_tol = 1e-6;

    //! Chord distance fudge factor
    static constexpr real_type dchord_tol = 1e-5 * units::millimeter;

    //! Lowest allowable scaling factor when searching for a chord
    static constexpr real_type min_chord_shrink = 0.5;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
      return  (minimum_step > 0)
	       && (delta_chord > 0)
	       && (delta_intersection > minimum_step)
	       && (epsilon_step > 0 && epsilon_step < 1)
	       && (epsilon_rel_max > 0)
	       && (pgrow < 0)
	       && (pshrink < 0)
	       && (safety > 0 && safety < 1)
	       && (max_stepping_increase > 1)
	       && (max_stepping_decrease > 0 && max_stepping_decrease < 1)
	       && (max_nsteps > 0) && (max_substeps > 0);
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
//! Equality operator
constexpr bool
operator==(FieldDriverOptions const& a, FieldDriverOptions const& b)
{
    // clang-format off
    return a.minimum_step == b.minimum_step
           && a.delta_chord == b.delta_chord
           && a.delta_intersection == b.delta_intersection
           && a.epsilon_step == b.epsilon_step
           && a.epsilon_rel_max == b.epsilon_rel_max
           && a.errcon == b.errcon
           && a.pgrow == b.pgrow
           && a.pshrink == b.pshrink
           && a.safety == b.safety
           && a.max_stepping_increase == b.max_stepping_increase
           && a.max_stepping_decrease == b.max_stepping_decrease
           && a.max_nsteps == b.max_nsteps
           && a.max_substeps == b.max_substeps
           && a.initial_step_tol == b.initial_step_tol
           && a.dchord_tol == b.dchord_tol
           && a.min_chord_shrink == b.min_chord_shrink;
    // clang-format on
}

//---------------------------------------------------------------------------//
// Throw a runtime assertion if any of the input is invalid
void validate_input(FieldDriverOptions const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas

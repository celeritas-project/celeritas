//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 */
struct FieldDriverOptions
{
    //! The minimum length of the field step
    real_type minimum_step = 1.0e-5 * units::millimeter;

    //! The closest miss distance
    real_type delta_chord = 0.25 * units::millimeter;

    //! Accuracy of intersection of the boundary crossing
    real_type delta_intersection = 1.0e-4 * units::millimeter;

    //! Relative error scale on the step length
    real_type epsilon_step = 1.0e-5;

    //! Maximum of the error ratio
    real_type epsilon_rel_max = 1.0e-3;

    //! Truncation error tolerance
    real_type errcon = 1.0e-4;

    //! Exponent to increase a step size
    real_type pgrow = -0.20;

    //! Exponent to decrease a step size
    real_type pshrink = -0.25;

    //! Scale factor for the predicted step size
    real_type safety = 0.9;

    //! Maximum scale to increase a step size
    real_type max_stepping_increase = 5;

    //! Maximum scale factor to decrease a step size
    real_type max_stepping_decrease = 0.1;

    //! Maximum number of steps (or trials)
    size_type max_nsteps = 100;

    //! Initial step tolerance
    static constexpr real_type initial_step_tol = 1e-6;

    //! Chord distance fudge factor
    static constexpr real_type dchord_tol = 1e-5 * units::millimeter;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
      return  (minimum_step > 0)
	       && (delta_chord > 0)
	       && (delta_intersection > minimum_step)
	       && (epsilon_step > 0 && epsilon_step < 1)
	       && (epsilon_rel_max > 0)
	       && (errcon > 0)
	       && (pgrow < 0)
	       && (pshrink < 0)
	       && (safety > 0 && safety < 1)
	       && (max_stepping_increase > 1)
	       && (max_stepping_decrease > 0 && max_stepping_decrease < 1)
	       && (max_nsteps > 0);
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

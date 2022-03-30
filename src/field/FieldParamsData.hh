//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldParamsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host and device parameters for propagation in a magnetic field and
 * suggested default values
 */
struct FieldParamsData
{
    //! the minimum length of the field step
    real_type minimum_step = 1.0e-5 * units::millimeter;

    //! the closest miss distrance
    real_type delta_chord = 0.25 * units::millimeter;

    //! accuracy of intersection of the boundary crossing
    real_type delta_intersection = 1.0e-4 * units::millimeter;

    //! the relative error scale on the step length
    real_type epsilon_step = 1.0e-5;

    //! the maximum of the error ratio
    real_type epsilon_rel_max = 1.0e-3;

    //! the truncation error tolerance
    real_type errcon = 1.0e-4;

    //! pgrow (the exponent to increase a step size)
    real_type pgrow = -0.20;

    //! pshrink (the exponent to decrease a step size)
    real_type pshrink = -0.25;

    //! safety (a scale factor for the predicted step size)
    real_type safety = 0.9;

    //! the maximum scale to increase a step size
    real_type max_stepping_increase = 5;

    //! the maximum scale factor to decrease a step size
    real_type max_stepping_decrease = 0.1;

    //! the maximum number of steps (or trials)
    size_type max_nsteps = 100;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        // clang-format off
      return  (minimum_step > 0)
	       && (delta_chord > 0)
	       && (delta_intersection > 0)
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

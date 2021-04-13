//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldParamsPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host and device parameters for propagation in a magnetic field and
 * suggested default values
 */
struct FieldParamsPointers
{
    //! the minimum length of the field step = 1.0e-5 [mm]
    real_type minimum_step;

    //! the closest miss distrance = 0.25 [mm]
    real_type delta_chord;

    //! accuracy of intersection of the boundary crossing  = 1.0e-4 [mm]
    real_type delta_intersection;

    //! the relative error scale on the step length = 1.0e-5
    real_type epsilon_step;

    //! the maximum of the error ratio = 1.0e-3
    real_type epsilon_rel_max;

    //! the truncation error tolerance = 1.0e-4
    real_type errcon;

    //! pgrow (the exponent to increase a stepsiz) = -0.20
    real_type pgrow;

    //! pshrink (the exponent to decrease a stepsiz) = -0.25
    real_type pshrink;

    //! safety (a scale factor for the predicted step size) = 0.9
    real_type safety;

    //! the maximum scale to increase a step size = 5
    real_type max_stepping_increase;

    //! the maximum scale factor to decrease a step size = 0.1
    real_type max_stepping_decrease;

    //! the maximum number of steps (or trials) = 100
    size_type max_nsteps;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return    minimum_step
               && delta_chord 
               && delta_intersection 
               && epsilon_step 
               && epsilon_rel_max 
               && errcon 
               && pgrow 
               && pshrink 
               && safety 
               && max_stepping_increase
               && max_stepping_decrease 
               && max_nsteps;
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

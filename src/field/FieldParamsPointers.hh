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
 * Host and device parameters for propagation in a magnetic field
 */
struct FieldParamsPointers
{
    //! accuracy for integration of one step = 0.01 mm
    //    real_type delta_one_step;

    //! the closest miss distrance = 0.25 mm
    real_type delta_chord;

    //! accuracy of intersection of a boundary crossing  = 0.0001 mm
    real_type delta_intersection;

    //! the relative error = 1.0e-5
    real_type epsilon_step;

    //! min step size = 1.0e-5
    real_type minimun_step;

    //! safety = 0.9
    real_type safety;

    //! pgrow = -0.20
    real_type pgrow;

    //! pshrink = -0.25
    real_type pshrink;

    //! errcon = 1.0e-4
    real_type errcon;

    //! max_stepping_increase = 5
    real_type max_stepping_increase;

    //! max_stepping_increase = 0.1
    real_type max_stepping_decrease;

    //! max_nsteps = 100
    size_type max_nsteps;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        // clang-format off
        return    delta_chord 
               && delta_intersection 
               && epsilon_step 
               && minimun_step 
               && safety 
               && pgrow 
               && pshrink 
               && errcon 
               && max_stepping_increase
               && max_stepping_decrease 
               && max_nsteps;
        // clang-format on
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

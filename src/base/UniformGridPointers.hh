//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformGridPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data needed to do uniform grid interpolation.
 */
struct UniformGridPointers
{
    size_type size;  //!< Number of grid edges/points
    real_type front; //!< Value of first grid point
    real_type delta; //!< Grid cell width

    CELER_FUNCTION explicit operator bool() const
    {
        return size > 0 && delta > 0;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

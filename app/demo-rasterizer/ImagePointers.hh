//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImagePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Construction arguments for an on-device image view.
 */
struct ImagePointers
{
    celeritas::Real3 origin;   //!< Upper left corner
    celeritas::Real3 down_ax;  //!< Downward axis (increasing j, track
                               //!< initialization)
    celeritas::Real3 right_ax; //!< Rightward axis (increasing i, track
                               //!< movement)
    celeritas::real_type              pixel_width; //!< Width of a pixel
    celeritas::array<unsigned int, 2> dims;        //!< Image dimensions (j, i)
    celeritas::span<int>              image;       //!< Stored image [j][i]

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return !image.empty(); }
};

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer

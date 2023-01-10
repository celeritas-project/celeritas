//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/ImageData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Construction arguments for an on-device image view.
 */
struct ImageData
{
    celeritas::Real3 origin;  //!< Upper left corner
    celeritas::Real3 down_ax;  //!< Downward axis (increasing j, track
                               //!< initialization)
    celeritas::Real3 right_ax;  //!< Rightward axis (increasing i, track
                                //!< movement)
    celeritas::real_type pixel_width;  //!< Width of a pixel
    celeritas::Array<unsigned int, 2> dims;  //!< Image dimensions (j, i)
    celeritas::Span<int> image;  //!< Stored image [j][i]

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return !image.empty(); }
};

//---------------------------------------------------------------------------//
}  // namespace demo_rasterizer

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/ImageData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construction arguments for an on-device image view.
 */
struct ImageData
{
    Real3 origin;  //!< Upper left corner
    Real3 down_ax;  //!< Downward axis (increasing j, track
                    //!< initialization)
    Real3 right_ax;  //!< Rightward axis (increasing i, track
                     //!< movement)
    real_type pixel_width;  //!< Width of a pixel
    Array<unsigned int, 2> dims;  //!< Image dimensions (j, i)
    Span<int> image;  //!< Stored image [j][i]

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return !image.empty(); }
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageTrackView.i.hh
//---------------------------------------------------------------------------//
#include "base/Assert.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION
ImageTrackView::ImageTrackView(const ImagePointers& shared, ThreadId tid)
    : shared_(shared), j_index_(tid.get())
{
    CELER_EXPECT(j_index_ < shared_.dims[0]);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate starting position.
 */
CELER_FUNCTION auto ImageTrackView::start_pos() const -> Real3
{
    Real3     result;
    real_type down_offset = (j_index_ + real_type(0.5)) * shared_.pixel_width;
    for (int i = 0; i < 3; ++i)
    {
        result[i] = shared_.origin[i] + shared_.down_ax[i] * down_offset;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set the value for a pixel.
 */
CELER_FUNCTION void ImageTrackView::set_pixel(unsigned int i, int value)
{
    CELER_EXPECT(i < shared_.dims[1]);
    unsigned int idx = j_index_ * shared_.dims[1] + i;

    CELER_ASSERT(idx < shared_.image.size());
    shared_.image[idx] = value;
}

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer

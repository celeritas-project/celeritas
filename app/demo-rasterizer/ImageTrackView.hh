//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"

#include "ImageData.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Modify a line of a rasterized image.
 *
 * The rasterizer starts at the left side of an image and traces rightward.
 * Each "line" is a single thread.
 */
class ImageTrackView
{
  public:
    //!@{
    //! Type aliases
    using ThreadId  = celeritas::ThreadId;
    using Real3     = celeritas::Real3;
    using real_type = celeritas::real_type;
    //!@}

  public:
    // Construct with image data and thread ID
    inline CELER_FUNCTION ImageTrackView(const ImageData& shared, ThreadId tid);

    // Calculate start position
    inline CELER_FUNCTION Real3 start_pos() const;

    //! Start direction (rightward axis)
    CELER_FUNCTION const Real3& start_dir() const { return shared_.right_ax; }

    //! Pixel width
    CELER_FUNCTION real_type pixel_width() const
    {
        return shared_.pixel_width;
    }

    // Set pixel value
    inline CELER_FUNCTION void set_pixel(unsigned int i, int value);

  private:
    const ImageData& shared_;
    unsigned int     j_index_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION
ImageTrackView::ImageTrackView(const ImageData& shared, ThreadId tid)
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

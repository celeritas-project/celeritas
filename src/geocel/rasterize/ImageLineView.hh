//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageLineView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/Types.hh"

#include "ImageData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Modify a line of a image for rasterization.
 *
 * The rasterizer starts at the left side of an image and traces rightward.
 * Each "line" is a single thread.
 *
 * \todo Add template parameter to switch from vertical to horizontal
 * direction.
 */
class ImageLineView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<ImageParamsData>;
    using StateRef = NativeRef<ImageStateData>;
    //!@}

  public:
    // Construct with image data and thread ID
    inline CELER_FUNCTION ImageLineView(ParamsRef const& params,
                                        StateRef const& state,
                                        size_type row_index);

    // Calculate start position
    inline CELER_FUNCTION Real3 start_pos() const;

    //! Start direction (rightward axis)
    CELER_FUNCTION Real3 const& start_dir() const { return scalars_.right; }

    //! Pixel width
    CELER_FUNCTION real_type pixel_width() const
    {
        return scalars_.pixel_width;
    }

    //! Maximum length to trace
    CELER_FUNCTION real_type max_length() const { return scalars_.max_length; }

    //! Number of pixels along the direction of travel
    CELER_FUNCTION size_type max_index() const { return scalars_.dims[1]; }

    // Set pixel value
    inline CELER_FUNCTION void set_pixel(size_type col, int value);

  private:
    ImageParamsScalars const& scalars_;
    StateRef const& state_;
    size_type row_index_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with image data and thread ID.
 */
CELER_FUNCTION
ImageLineView::ImageLineView(ParamsRef const& params,
                             StateRef const& state,
                             size_type row_index)
    : scalars_{params.scalars}, state_{state}, row_index_{row_index}
{
    CELER_EXPECT(row_index_ < scalars_.dims[0]);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate starting position.
 */
CELER_FUNCTION auto ImageLineView::start_pos() const -> Real3
{
    real_type down_offset = (row_index_ + real_type(0.5))
                            * scalars_.pixel_width;
    Real3 result = scalars_.origin;
    axpy(down_offset, scalars_.down, &result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set the value for a pixel.
 */
CELER_FUNCTION void ImageLineView::set_pixel(size_type col, int value)
{
    CELER_EXPECT(col < scalars_.dims[1]);
    size_type idx = row_index_ * scalars_.dims[1] + col;

    CELER_ASSERT(idx < state_.image.size());
    state_.image[ItemId<int>{idx}] = value;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ImageInterface.hh"

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
    inline CELER_FUNCTION
    ImageTrackView(const ImagePointers& shared, ThreadId tid);

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
    const ImagePointers& shared_;
    unsigned int         j_index_;
};

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer

#include "ImageTrackView.i.hh"

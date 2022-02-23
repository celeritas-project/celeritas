//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "base/Array.hh"
#include "base/DeviceVector.hh"
#include "base/Span.hh"
#include "base/Types.hh"

#include "ImageData.hh"
#include "ImageIO.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Initialization and storage for a raster image.
 */
class ImageStore
{
  public:
    //!@{
    //! Type aliases
    using real_type = celeritas::real_type;
    using UInt2     = celeritas::Array<unsigned int, 2>;
    using Real3     = celeritas::Real3;
    using VecInt    = std::vector<int>;
    //!@}

  public:
    // Construct with defaults
    explicit ImageStore(ImageRunArgs);

    //// DEVICE ACCESSORS ////

    //! Access image on host for initializing
    ImageData host_interface();

    //! Access image on device for writing
    ImageData device_interface();

    //// HOST ACCESSORS ////

    //! Upper left corner of the image
    const Real3& origin() const { return origin_; }

    //! Downward axis (increasing j)
    const Real3& down_ax() const { return down_ax_; }

    //! Rightward axis (increasing i)
    const Real3& right_ax() const { return right_ax_; }

    //! Width of a pixel
    real_type pixel_width() const { return pixel_width_; }

    //! Dimensions {j, i} of the image
    const UInt2& dims() const { return dims_; }

    // Copy out the image to the host
    VecInt data_to_host() const;

  private:
    Real3                        origin_;
    Real3                        down_ax_;
    Real3                        right_ax_;
    real_type                    pixel_width_;
    UInt2                        dims_;
    celeritas::DeviceVector<int> image_;
};

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer

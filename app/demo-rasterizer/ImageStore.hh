//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/ImageStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/DeviceVector.hh"
#include "celeritas/Types.hh"

#include "ImageData.hh"
#include "ImageIO.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Initialization and storage for a raster image.
 */
class ImageStore
{
  public:
    //!@{
    //! \name Type aliases

    using UInt2 = Array<unsigned int, 2>;

    using VecInt = std::vector<int>;
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
    Real3 const& origin() const { return origin_; }

    //! Downward axis (increasing j)
    Real3 const& down_ax() const { return down_ax_; }

    //! Rightward axis (increasing i)
    Real3 const& right_ax() const { return right_ax_; }

    //! Width of a pixel
    real_type pixel_width() const { return pixel_width_; }

    //! Dimensions {j, i} of the image
    UInt2 const& dims() const { return dims_; }

    // Copy out the image to the host
    VecInt data_to_host() const;

  private:
    Real3 origin_;
    Real3 down_ax_;
    Real3 right_ax_;
    real_type pixel_width_;
    UInt2 dims_;
    DeviceVector<int> image_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

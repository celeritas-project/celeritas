//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageStore.cc
//---------------------------------------------------------------------------//
#include "ImageStore.hh"

#include "base/ArrayUtils.hh"
#include "base/Range.hh"

using celeritas::range;

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Construct with image slice and extents.
 */
ImageStore::ImageStore(ImageRunArgs params)
{
    REQUIRE(celeritas::is_soft_unit_vector(params.rightward_ax,
                                           celeritas::SoftEqual<real_type>{}));
    REQUIRE(params.lower_left != params.upper_right);
    REQUIRE(params.vertical_pixels > 0);

    // Normalize rightward axis
    right_ax_ = params.rightward_ax;
    celeritas::normalize_direction(&right_ax_);

    // Vector pointing toward the upper right from the lower left corner
    Real3 diagonal;
    for (int i : range(3))
    {
        diagonal[i] = params.upper_right[i] - params.lower_left[i];
    }

    // Set downward axis to the diagonal with the rightward component
    // subtracted out; then normalize
    real_type projection = celeritas::dot_product(diagonal, right_ax_);
    for (int i : range(3))
    {
        down_ax_[i] = -diagonal[i] + projection * right_ax_[i];
    }
    celeritas::normalize_direction(&down_ax_);

    // Calculate length along each axis
    real_type width_x = celeritas::dot_product(diagonal, right_ax_);
    real_type width_y = -celeritas::dot_product(diagonal, down_ax_);

    CHECK(width_x > 0 && width_y > 0);

    // Set number of pixels in each direction.
    auto num_y   = params.vertical_pixels;
    pixel_width_ = width_y / num_y;
    auto num_x = std::max<unsigned int>(std::ceil(width_x / pixel_width_), 1);

    // Set upper left corner
    for (int i : range(3))
    {
        origin_[i] = params.lower_left[i] - down_ax_[i] * num_y * pixel_width_;
    }

    // Allocate storage
    dims_  = {num_y, num_x};
    image_ = celeritas::DeviceVector<int>(num_y * num_x);
    ENSURE(!image_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Access image on host for initializing.
 */
ImagePointers ImageStore::host_interface()
{
    ImagePointers result;

    result.origin      = origin_;
    result.down_ax     = down_ax_;
    result.right_ax    = right_ax_;
    result.pixel_width = pixel_width_;
    result.dims        = dims_;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access image on device for writing.
 */
ImagePointers ImageStore::device_interface()
{
    ImagePointers result;

    result.origin      = origin_;
    result.down_ax     = down_ax_;
    result.right_ax    = right_ax_;
    result.pixel_width = pixel_width_;
    result.dims        = dims_;
    result.image       = image_.device_pointers();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Copy out the image to the host.
 */
auto ImageStore::data_to_host() const -> VecInt
{
    VecInt result(dims_[0] * dims_[1]);
    image_.copy_to_host(celeritas::make_span(result));
    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer

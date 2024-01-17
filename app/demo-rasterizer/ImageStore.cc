//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/ImageStore.cc
//---------------------------------------------------------------------------//
#include "ImageStore.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with image slice and extents.
 */
ImageStore::ImageStore(ImageRunArgs params)
{
    CELER_EXPECT(is_soft_unit_vector(params.rightward_ax));
    CELER_EXPECT(params.lower_left != params.upper_right);
    CELER_EXPECT(params.vertical_pixels > 0);

    // Normalize rightward axis
    right_ax_ = make_unit_vector(params.rightward_ax);

    // Vector pointing toward the upper right from the lower left corner
    Real3 diagonal;
    for (int i : range(3))
    {
        diagonal[i] = params.upper_right[i] - params.lower_left[i];
    }

    // Set downward axis to the diagonal with the rightward component
    // subtracted out; then normalize
    real_type projection = dot_product(diagonal, right_ax_);
    for (int i : range(3))
    {
        down_ax_[i] = -diagonal[i] + projection * right_ax_[i];
    }
    down_ax_ = make_unit_vector(down_ax_);

    // Calculate length along each axis
    real_type width_x = dot_product(diagonal, right_ax_);
    real_type width_y = -dot_product(diagonal, down_ax_);

    CELER_ASSERT(width_x > 0 && width_y > 0);

    // Set number of pixels in each direction.
    auto num_y = params.vertical_pixels;
    pixel_width_ = width_y / num_y;
    auto num_x = std::max<unsigned int>(std::ceil(width_x / pixel_width_), 1);

    // Set upper left corner
    for (int i : range(3))
    {
        origin_[i] = params.lower_left[i] - down_ax_[i] * num_y * pixel_width_;
    }

    // Allocate storage
    dims_ = {num_y, num_x};
    image_ = DeviceVector<int>(num_y * num_x);
    CELER_ENSURE(!image_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Access image on host for initializing.
 */
ImageData ImageStore::host_interface()
{
    ImageData result;

    result.origin = origin_;
    result.down_ax = down_ax_;
    result.right_ax = right_ax_;
    result.pixel_width = pixel_width_;
    result.dims = dims_;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access image on device for writing.
 */
ImageData ImageStore::device_interface()
{
    ImageData result;

    result.origin = origin_;
    result.down_ax = down_ax_;
    result.right_ax = right_ax_;
    result.pixel_width = pixel_width_;
    result.dims = dims_;
    result.image = image_.device_ref();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Copy out the image to the host.
 */
auto ImageStore::data_to_host() const -> VecInt
{
    VecInt result(dims_[0] * dims_[1]);
    image_.copy_to_host(make_span(result));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

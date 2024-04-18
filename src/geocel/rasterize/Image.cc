//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Image.cc
//---------------------------------------------------------------------------//
#include "Image.hh"

#include <cmath>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with image properties.
 *
 * All inputs should be in the native unit system.
 */
ImageParams::ImageParams(ImageInput&& inp)
{
    CELER_VALIDATE(ArraySoftUnit{real_type{0.001}}(inp.rightward),
                   << "rightward axis " << repr(inp.rightward)
                   << " is not a unit vector");
    CELER_VALIDATE(inp.lower_left != inp.upper_right,
                   << "lower left corner " << repr(inp.lower_left)
                   << " and upper right corner cannot be the same");
    CELER_VALIDATE(inp.vertical_pixels > 0,
                   << "number of pixels must be positive");
    CELER_VALIDATE(inp.horizontal_divisor > 0,
                   << "number of horizontal chunks must be positive");

    ImageParamsScalars scalars;

    // Normalize rightward axis
    scalars.right = make_unit_vector(inp.rightward);

    // Vector pointing toward the upper right from the lower left corner
    Real3 diagonal = inp.upper_right - inp.lower_left;

    // Set downward axis to the diagonal with the rightward component
    // subtracted out; then normalize
    real_type projection = dot_product(diagonal, scalars.right);
    scalars.down = -diagonal;
    axpy(projection, scalars.right, &scalars.down);
    scalars.down = make_unit_vector(scalars.down);

    // Calculate length along each axis
    real_type width_x = dot_product(diagonal, scalars.right);
    real_type width_y = -dot_product(diagonal, scalars.down);
    CELER_ASSERT(width_x > 0 && width_y > 0);
    scalars.max_length = width_x;

    // Set number of pixels in each direction.
    size_type num_y = inp.vertical_pixels;
    scalars.pixel_width = width_y / num_y;
    size_type num_x
        = inp.horizontal_divisor
          * std::ceil(width_x / (inp.horizontal_divisor * scalars.pixel_width));
    CELER_ASSERT(num_x >= inp.horizontal_divisor);
    scalars.dims = {num_y, num_x};

    // Set upper left corner
    scalars.origin = inp.lower_left;
    axpy(-(num_y * scalars.pixel_width), scalars.down, &scalars.origin);

    // Allocate storage and "copy" to device
    CELER_ASSERT(scalars);
    data_
        = CollectionMirror<ImageParamsData>{HostVal<ImageParamsData>{scalars}};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from parameters.
 */
template<MemSpace M>
Image<M>::Image(SPConstParams params) : params_(std::move(params))
{
    CELER_EXPECT(params_);

    // Allocate the image, save a reference, and fill with "invalid"
    resize(&value_, params_->host_ref());
    ref_ = value_;
    celeritas::fill(-1, &ref_.image);
}

//---------------------------------------------------------------------------//
/*!
 * Copy the image back to the host.
 */
template<MemSpace M>
void Image<M>::copy_to_host(SpanInt out) const
{
    CELER_VALIDATE(out.size() == ref_.image.size(),
                   << "invalid output size " << out.size()
                   << " for image copying: should be " << ref_.image.size());
    celeritas::copy_to_host(ref_.image, out);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Image<MemSpace::host>;
template class Image<MemSpace::device>;

//---------------------------------------------------------------------------//
}  // namespace celeritas

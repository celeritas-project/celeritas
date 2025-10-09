//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Image.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "ImageData.hh"
#include "ImageInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Image construction arguments.
 *
 * Image scale in this struct is *native* units, but JSON I/O defaults to
 * centimeters for the window coordinates and accepts an optional "_units"
 * parameter that can take values of cgs, si, or clhep to interpret the input
 * as centimeters, meters, or millimeters.
 */
struct ImageInput
{
    //!@{
    //! Coordinates of the window [length]
    Real3 lower_left{0, 0, 0};
    Real3 upper_right{};
    //!@}

    //! Rightward basis vector, the new "x" axis
    Real3 rightward{1, 0, 0};

    //! Number of vertical pixels, aka threads when raytracing
    size_type vertical_pixels{};

    //! Round the number of horizontal pixels to this value
    size_type horizontal_divisor{CELER_USE_DEVICE ? 128 / sizeof(int) : 1};

    //! True if the input is unassigned
    explicit operator bool() const
    {
        return vertical_pixels != 0 && lower_left != upper_right;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Manage properties of an image.
 *
 * An image is a "window", a 2D rectangle slice of 3D space. As with computer
 * GUI windows, matplotlib \c imshow, and other visual rendering layouts, the
 * pixel order is like text on a page: left to right, then top to bottom.
 * Because this is vertically flipped from "mathematical" ordering, we store
 * the upper left coordinate and a \em -y basis vector rather than a lower left
 * coordinate and a \em +y basis vector.
 *
 * The same image params can be used to construct multiple images (using
 * different ray tracing methods or different geometries or on host vs device).
 */
class ImageParams final : public ParamsDataInterface<ImageParamsData>
{
  public:
    // Construct with image properties
    explicit ImageParams(ImageInput const&);

    //! Access scalar image properties
    ImageParamsScalars const& scalars() const
    {
        return this->host_ref().scalars;
    }

    //! Number of pixels in an image created from these params
    size_type num_pixels() const
    {
        auto const& dims = this->scalars().dims;
        return dims[0] * dims[1];
    }

    //! Number of horizontal lines to be used for raytracing
    size_type num_lines() const { return this->scalars().dims[0]; }

    //! Access properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<ImageParamsData> data_;
};

//---------------------------------------------------------------------------//
/*!
 * Implement an image on host or device.
 */
template<MemSpace M>
class Image final : public ImageInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Value = ImageStateData<Ownership::value, M>;
    using Ref = ImageStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct from parameters
    explicit Image(SPConstParams params);

    //! Access image properties
    SPConstParams const& params() const final { return params_; }

    // Write the image to a stream in binary format
    void copy_to_host(SpanInt) const final;

    //! Access the mutable state data
    Ref const& ref() { return ref_; }

  private:
    SPConstParams params_;
    Value value_;
    Ref ref_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

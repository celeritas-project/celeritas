//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Image.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "ImageData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Image construction arguments
struct ImageInput
{
    Real3 lower_left{0, 0, 0};
    Real3 upper_right{};

    //! Rightward basis vector, the new "x" axis
    Real3 rightward{1, 0, 0};

    //! Number of vertical pixels, aka threads when raytracing
    size_type vertical_pixels{};

    //! Round the number of horizontal pixels to this value
    size_type horizontal_divisor{CELER_USE_DEVICE ? 128 / sizeof(int) : 1};
};

//---------------------------------------------------------------------------//
/*!
 * Manage properties of an image.
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
 * Access data from an image.
 */
class ImageInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SpanInt = Span<int>;
    using SPConstParams = std::shared_ptr<ImageParams const>;
    //!@}

  public:
    //! Access image properties
    virtual SPConstParams const& params() const = 0;

    //! Copy the image to the host
    virtual void copy_to_host(SpanInt) const = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~ImageInterface() = default;

    ImageInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ImageInterface);
};

//---------------------------------------------------------------------------//
/*!
 * Implement an image on host or device.
 */
template<MemSpace M>
class Image : public ImageInterface
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

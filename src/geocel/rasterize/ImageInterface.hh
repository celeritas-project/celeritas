//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
class ImageParams;

//---------------------------------------------------------------------------//
/*!
 * Access data from an image.
 *
 * Images currently are arrays of integer pixels.
 */
class ImageInterface
{
  public:
    //!@{
    //! \name Type aliases
    using int_type = int;
    using SpanInt = Span<int_type>;
    using SPConstParams = std::shared_ptr<ImageParams const>;
    //!@}

  public:
    //! Default virtual destructor
    virtual ~ImageInterface() = default;

    //! Access image properties
    virtual SPConstParams const& params() const = 0;

    //! Copy the image to the host
    virtual void copy_to_host(SpanInt) const = 0;

  protected:
    ImageInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ImageInterface);
};

// Forward-declare image, see Image.hh
template<MemSpace M>
class Image;

//---------------------------------------------------------------------------//
/*!
 * Generate one or more images from a geometry.
 */
class ImagerInterface
{
  public:
    //! Default virtual destructor
    virtual ~ImagerInterface() = default;

    //!@{
    //! Raytrace an image on host or device
    virtual void operator()(Image<MemSpace::host>*) = 0;
    virtual void operator()(Image<MemSpace::device>*) = 0;
    //!@}

  protected:
    ImagerInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ImagerInterface);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

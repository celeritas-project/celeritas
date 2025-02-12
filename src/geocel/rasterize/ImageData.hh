//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Scalar properties for building a rasterized image.
 *
 * These properties specify a "window" that's a slice of a 3D geometry. It uses
 * graphics conventions of making the upper left corner the origin.
 *
 * The \c down basis vector corresponds to increasing \em j and is used for
 * track initialization.  The \c right basis vector corresponds to increasing
 * \em i and is used for track movement. Because the user-specified window may
 * not have an integer ratio of the two sides, we have a "max length" for
 * raytracing to the right. This also lets us round up the image dimensions to
 * a convenient alignment.
 *
 * All units are "native" length.
 */
struct ImageParamsScalars
{
    Real3 origin{};  //!< Upper left corner
    Real3 down{};  //!< Downward basis vector
    Real3 right{};  //!< Rightward basis vector (increasing i, track movement)
    real_type pixel_width{};  //!< Width of a pixel
    Size2 dims{};  //!< Image dimensions (row, column) = (y, x)
    real_type max_length{};  //!< Maximum distance along rightward to trace

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return pixel_width > 0 && dims[0] > 0 && dims[1] > 0 && max_length > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data used to construct an image.
 *
 * TODO: add material/cell -> RGB for inline rendering?
 */
template<Ownership W, MemSpace M>
struct ImageParamsData
{
    //// DATA ////

    ImageParamsScalars scalars;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(scalars);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ImageParamsData& operator=(ImageParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Image state data.
 *
 * This is just a representation of the image itself.
 */
template<Ownership W, MemSpace M>
struct ImageStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    //// DATA ////

    Items<int> image;  //!< Stored image [i][j]

    //// METHODS ////

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const { return !image.empty(); }

    //! Number of pixels
    CELER_FUNCTION size_type size() const { return image.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ImageStateData& operator=(ImageStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        image = other.image;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize geometry tracking states.
 */
template<MemSpace M>
inline void resize(ImageStateData<Ownership::value, M>* data,
                   HostCRef<ImageParamsData> const& params)
{
    CELER_EXPECT(data);
    CELER_EXPECT(params);

    resize(&data->image, params.scalars.dims[0] * params.scalars.dims[1]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

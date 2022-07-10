//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/SurfaceInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "orange/Data.hh"
#include "orange/Types.hh"

#include "SurfaceInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct surfaces on the host.
 *
 * Currently this simply appends the surface to the Data, but the full
 * robust geometry implementation will implement "soft" surface deduplication.
 *
 * \code
   SurfaceInserter insert_surface(&params.surfaces);
   auto id = insert_surface(PlaneX(123));
   auto id2 = insert_surface(PlaneX(123.0000001)); // TODO: equals id
   \endcode
 */
class SurfaceInserter
{
  public:
    //!@{
    //! Type aliases
    using Data         = HostVal<SurfaceData>;
    using SurfaceRange = ItemRange<struct Surface>;
    //!@}

    //! Type-deleted reference to a surface
    struct GenericSurfaceRef
    {
        SurfaceType           type;
        Span<const real_type> data;

        inline explicit operator bool() const;
    };

  public:
    // Construct with reference to surfaces to build
    explicit SurfaceInserter(Data* surfaces);

    // Add a new surface
    template<class T>
    inline SurfaceId operator()(const T& surface);

    // Append a generic surface view to the vector
    SurfaceId operator()(GenericSurfaceRef generic_surf);

    // Create a bunch of surfaces (experimental)
    SurfaceRange operator()(const SurfaceInput& all_surfaces);

  private:
    Data* surface_data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether a generic surface reference is valid.
 */
SurfaceInserter::GenericSurfaceRef::operator bool() const
{
    return !data.empty() && type != SurfaceType::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Add a surface (type-deleted) with the given coefficients
 */
template<class T>
SurfaceId SurfaceInserter::operator()(const T& surface)
{
    static_assert(sizeof(typename T::Intersections) > 0,
                  "Template parameter must be a surface class");

    return (*this)(GenericSurfaceRef{surface.surface_type(), surface.data()});
}

//---------------------------------------------------------------------------//
} // namespace celeritas

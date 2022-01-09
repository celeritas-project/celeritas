//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Surfaces.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../Data.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access stored surface data.
 *
 * These are all surfaces in the entire geometry.
 */
class Surfaces
{
  public:
    //@{
    //! Type aliases
    using SurfaceDataRef
        = SurfaceData<Ownership::const_reference, MemSpace::native>;
    //@}

  public:
    // Construct with reference to persistent data
    explicit inline CELER_FUNCTION Surfaces(const SurfaceDataRef&);

    // Number of surfaces
    inline CELER_FUNCTION SurfaceId::size_type num_surfaces() const;

    // Get the type of an indexed surface
    inline CELER_FUNCTION SurfaceType surface_type(SurfaceId) const;

    // Convert a stored surface to a class
    template<class T>
    inline CELER_FUNCTION T make_surface(SurfaceId) const;

  private:
    const SurfaceDataRef& data_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent data.
 */
CELER_FUNCTION Surfaces::Surfaces(const SurfaceDataRef& data) : data_(data)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Number of surfaces.
 */
CELER_FUNCTION SurfaceId::size_type Surfaces::num_surfaces() const
{
    return data_.types.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the type of an indexed surface.
 */
CELER_FUNCTION SurfaceType Surfaces::surface_type(SurfaceId sid) const
{
    CELER_EXPECT(sid < this->num_surfaces());
    return data_.types[sid];
}

//---------------------------------------------------------------------------//
/*!
 * Convert a stored surface to a class.
 */
template<class T>
CELER_FUNCTION T Surfaces::make_surface(SurfaceId sid) const
{
    static_assert(T::Storage::extent > 0,
                  "Template parameter must be a surface class");
    CELER_EXPECT(sid < this->num_surfaces());
    CELER_EXPECT(this->surface_type(sid) == T::surface_type());

    OpaqueId<real_type> start = data_.offsets[sid];
    OpaqueId<real_type> end{start.get()
                            + static_cast<size_type>(T::Storage::extent)};
    CELER_ASSERT(end.unchecked_get() <= data_.reals.size());
    return T{data_.reals[ItemRange<real_type>{start, end}]};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

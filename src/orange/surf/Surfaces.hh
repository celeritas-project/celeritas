//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Surfaces.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Data.hh"

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
    using ParamsRef = NativeCRef<OrangeParamsData>;
    //@}

  public:
    // Construct with reference to persistent data
    inline CELER_FUNCTION
    Surfaces(const ParamsRef& params, const SurfacesRecord& surf_record);

    // Number of surfaces
    inline CELER_FUNCTION SurfaceId::size_type num_surfaces() const;

    // Get the type of an indexed surface
    inline CELER_FUNCTION SurfaceType surface_type(SurfaceId) const;

    // Convert a stored surface to a class
    template<class T>
    inline CELER_FUNCTION T make_surface(SurfaceId) const;

  private:
    const ParamsRef&      params_;
    const SurfacesRecord& surf_record_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent data.
 */
CELER_FUNCTION
Surfaces::Surfaces(const ParamsRef& params, const SurfacesRecord& surf_record)
    : params_(params), surf_record_(surf_record)
{
}

//---------------------------------------------------------------------------//
/*!
 * Number of surfaces.
 */
CELER_FUNCTION SurfaceId::size_type Surfaces::num_surfaces() const
{
    return surf_record_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get the type of an indexed surface.
 */
CELER_FUNCTION SurfaceType Surfaces::surface_type(SurfaceId sid) const
{
    CELER_EXPECT(sid < this->num_surfaces());
    OpaqueId<SurfaceType> offset = *surf_record_.types.begin();
    CELER_ASSERT(sid.unchecked_get() + offset.unchecked_get()
                 < params_.surface_types.size());
    return params_.surface_types[offset + sid.unchecked_get()];
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
    CELER_EXPECT(sid < surf_record_.data_offsets.size());
    CELER_EXPECT(this->surface_type(sid) == T::surface_type());

    const real_type* data = params_.reals[AllItems<real_type>{}].data();

    OpaqueId<real_type> start_offset
        = params_.real_ids[surf_record_.data_offsets[sid.unchecked_get()]];
    auto stop_offset = start_offset.unchecked_get()
                       + static_cast<size_type>(T::Storage::extent);
    CELER_ASSERT(stop_offset <= params_.reals.size());
    return T{Span<const real_type>{data + start_offset.unchecked_get(),
                                   data + stop_offset}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

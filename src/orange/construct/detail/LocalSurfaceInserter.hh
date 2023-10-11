//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/LocalSurfaceInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "orange/OrangeTypes.hh"
#include "orange/surf/SoftSurfaceEqual.hh"
#include "orange/surf/VariantSurface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Merge local surfaces as they're being built.
 *
 * This will *always* insert the surface, but *sometimes* will return the ID of
 * a previously inserted surface.
 */
class LocalSurfaceInserter
{
  public:
    //!@{
    //! \name Type aliases
    using VecSurface = std::vector<VariantSurface>;
    //!@}

  public:
    // Construct with tolerance and a pointer to the surfaces vector
    LocalSurfaceInserter(VecSurface* v, Tolerance<> const& tol);

    // Construct a surface with deduplication
    template<class S>
    inline LocalSurfaceId operator()(S const& surface);

  private:
    using MapSurfId = std::unordered_map<LocalSurfaceId, LocalSurfaceId>;

    VecSurface* surfaces_;
    SoftSurfaceEqual soft_surface_equal_;
    ExactSurfaceEqual exact_surface_equal_;
    MapSurfId merged_;

    LocalSurfaceId merge_impl(LocalSurfaceId source, LocalSurfaceId target);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a surface with deduplication.
 *
 * There are three cases to consider:
 * - The new surface is entirely unique (insert, return new ID)
 * - The surface is soft equivalent but not exactly like an existing surface
 *   (insert, return existing ID)
 * - The surface is exactly the same (don't insert, return existing id)
 */
template<class S>
LocalSurfaceId LocalSurfaceInserter::operator()(S const& source)
{
    // Default result: new ID, add the surface
    LocalSurfaceId result(surfaces_->size());
    bool do_insert{true};

    // Test for soft equality against all existing surfaces
    // TODO: accelerate by mapping surfaces into a hash and comparing all
    // neighboring hash cells
    for (auto i : range(result))
    {
        VariantSurface const& vtarget = (*surfaces_)[i.unchecked_get()];
        if (S const* target = std::get_if<S>(&vtarget))
        {
            if (soft_surface_equal_(source, *target))
            {
                if (exact_surface_equal_(source, *target))
                {
                    // Surfaces are *exactly* equal:
                    result = i;
                    do_insert = false;
                }
                else
                {
                    // Surfaces are *nearly* equal
                    result = this->merge_impl(result, i);
                }
                break;
            }
        }
    }

    if (do_insert)
    {
        // Copy the new surface to the vector
        surfaces_->emplace_back(std::in_place_type<S>, source);
    }

    CELER_ENSURE(result < surfaces_->size());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

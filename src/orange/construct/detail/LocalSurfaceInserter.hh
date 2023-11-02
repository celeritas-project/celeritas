//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/LocalSurfaceInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
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
 * This will \em sometimes will return the ID of a previously inserted surface,
 * and \em sometimes will push the surface onto the vector of existing ones.
 *
 * There are three cases to consider:
 * - The new surface is entirely unique: we insert and return the new ID.
 * - The surface is soft equivalent but not exactly like an existing surface:
 *   we insert but return an existing ID.
 * - The surface is exactly the same: we do \em not insert, and return existing
 *   id.
 *
 * The second case adds the surface so that multiple nearby surfaces can be
 * \em chained together, even if the tolerance between the furthest apart is
 * greater than the soft equivalence tolerance.
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
    //// TYPES ////

    using MapSurfId = std::unordered_map<LocalSurfaceId, LocalSurfaceId>;

    //// DATA ////

    VecSurface* surfaces_;
    SoftSurfaceEqual soft_surface_equal_;
    ExactSurfaceEqual exact_surface_equal_;
    MapSurfId merged_;

    //// METHODS ////

    LocalSurfaceId merge_impl(LocalSurfaceId source, LocalSurfaceId target);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a surface with deduplication.
 */
template<class S>
LocalSurfaceId LocalSurfaceInserter::operator()(S const& source)
{
    VecSurface& all_surf = *surfaces_;

    // Test for soft equality against all existing surfaces
    auto is_soft_equal = [this, &source](VariantSurface const& vtarget) {
        if (S const* target = std::get_if<S>(&vtarget))
        {
            return soft_surface_equal_(source, *target);
        }
        return false;
    };

    // TODO: instead of linear search (making overall unit insertion
    // quadratic!) accelerate by mapping surfaces into a hash and comparing all
    // neighboring hash cells
    auto iter = std::find_if(all_surf.begin(), all_surf.end(), is_soft_equal);
    if (iter == all_surf.end())
    {
        // Surface is completely unique
        LocalSurfaceId result(all_surf.size());
        all_surf.emplace_back(std::in_place_type<S>, source);
        return result;
    }

    // Surface is soft equivalent to an existing surface at this index
    LocalSurfaceId target_id{static_cast<size_type>(iter - all_surf.begin())};

    CELER_ASSUME(std::holds_alternative<S>(*iter));
    if (exact_surface_equal_(source, std::get<S>(*iter)))
    {
        // Surfaces are *exactly* equal: don't insert
        CELER_ENSURE(target_id < all_surf.size());
        return target_id;
    }

    // Surface is a little bit different, so we still need to insert it
    // to chain duplicates
    LocalSurfaceId source_id(all_surf.size());
    all_surf.emplace_back(std::in_place_type<S>, source);

    // Store the equivalency relationship and potentially chain equivalent
    // surfaces
    return this->merge_impl(source_id, target_id);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

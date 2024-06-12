//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LocalSurfaceInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "orange/OrangeTypes.hh"
#include "orange/surf/SoftSurfaceEqual.hh"
#include "orange/surf/VariantSurface.hh"

#include "SurfaceGridHash.hh"

namespace celeritas
{
namespace orangeinp
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
 * - The surface is exactly the same: we do \em not insert, and return the
 *   existing id.
 *
 * The second case adds the surface so that multiple nearby surfaces can be
 * \em chained together, even if the tolerance between the furthest apart is
 * greater than the soft equivalence tolerance.
 *
 * To speed up insertion when large numbers of potentially similar surface are
 * constructed, this class uses a hash table to "bin" surfaces based on a
 * function of their spatial coordinates. Surfaces that *could* compare equal
 * all share the same bin or are at the edge of an adjacent one. Since this
 * hash table uses the standard library's implementation, it resizes
 * dynamically to keep the number of surfaces per bucket low (assuming a good
 * hash function), thus keeping insertion time at an amortized O(1) versus the
 * O(N) that would result from comparing against every other surface.
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
    LocalSurfaceId operator()(S const& surface);

  private:
    //// TYPES ////

    using MapSurfId = std::unordered_map<LocalSurfaceId, LocalSurfaceId>;
    using MultimapSurfaces
        = std::unordered_multimap<SurfaceGridHash::key_type, LocalSurfaceId>;

    //// DATA ////

    VecSurface* surfaces_;

    // Deduplication
    SoftSurfaceEqual soft_surface_equal_;
    ExactSurfaceEqual exact_surface_equal_;
    MapSurfId merged_;

    // Hash acceleration
    MultimapSurfaces hashed_surfaces_;
    SurfaceGridHash calc_hashes_;

    //// METHODS ////

    LocalSurfaceId merge_impl(LocalSurfaceId source, LocalSurfaceId target);
    LocalSurfaceId find_merged(LocalSurfaceId target) const;

    //! Width of a hash bin, in units of the problem's length scale
    static constexpr real_type bin_width_frac() { return 0.01; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas

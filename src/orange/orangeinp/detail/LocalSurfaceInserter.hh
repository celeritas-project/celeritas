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
    LocalSurfaceId operator()(S const& surface);

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
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas

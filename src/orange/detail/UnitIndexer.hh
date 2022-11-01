//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitIndexer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unit input to params data.
 *
 * Linearize the data in a UnitInput and add it to the host.
 */
class UnitIndexer
{
  public:
    //!@{
    //! \name Type aliases
    using LocalSurface = std::tuple<UniverseId, SurfaceId>;
    using LocalVolume  = std::tuple<UniverseId, VolumeId>;
    using VecSize      = std::vector<size_type>;
    //!@}

  public:
    // Construct from sizes
    UnitIndexer(VecSize num_surfaces, VecSize num_volumes);

    // Local-to-global
    SurfaceId global_surface(UniverseId uni, SurfaceId surface) const;
    VolumeId  global_volume(UniverseId uni, VolumeId volume) const;

    // Global-to-local
    LocalSurface local_surface(SurfaceId id) const;
    LocalVolume  local_volume(VolumeId id) const;

    //! Total number of universes
    size_type num_universes() const { return surfaces_.size() - 1; }

    //! Total number of surfaces
    size_type num_surfaces() const { return surfaces_.back(); }

    //! Total number of cells
    size_type num_volumes() const { return volumes_.back(); }

  private:
    //// DATA ////
    VecSize surfaces_;
    VecSize volumes_;

    //// IMPLEMENTATION METHODS ////
    static inline VecSize::const_iterator
    find_local(const VecSize& offsets, size_type id);

    static inline size_type local_size(const VecSize& offsets, UniverseId uni);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"
#include "celeritas/inp/Grid.hh"
#include "celeritas/inp/SurfacePhysics.hh"

// Geant4 forward declaration
class G4OpticalSurface;  // IWYU pragma: keep
class G4MaterialPropertiesTable;  // IWYU pragma: keep

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class used by \c GeantSurfacePhysicsLoader .
 */
class GeantSurfacePhysicsHelper
{
  public:
    // Construct with SurfaceId; this expects a valid GeantGeoParams
    GeantSurfacePhysicsHelper(SurfaceId sid);

    // Get optical surface id
    SurfaceId surface_id() const { return sid_; }

    // Get Geant4 optical surface
    G4OpticalSurface const& surface() const;

    // Populate Grid optical property from name, in [MeV, unitless]
    bool get_property(inp::Grid* dst, std::string const& name) const;

    // Insert a value into a map in place
    template<class T>
    inline void emplace(std::map<SurfaceId, T>& m, T&& value) const;

  private:
    SurfaceId sid_;
    G4OpticalSurface const* surface_;
    G4MaterialPropertiesTable const* mpt_;
};

//---------------------------------------------------------------------------//
/*!
 * Insert a value into a map for the current surface.
 */
template<class T>
inline void
GeantSurfacePhysicsHelper::emplace(std::map<SurfaceId, T>& m, T&& value) const
{
    auto result = m.emplace(sid_, std::forward<T>(value));
    // Duplicate surfaces are prohibited
    CELER_ASSERT(result.second);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

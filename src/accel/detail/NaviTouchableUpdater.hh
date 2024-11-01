//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/NaviTouchableUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "geocel/GeantGeoUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

class G4Navigator;
class G4VPhysicalVolume;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Update the temporary navigation state based on the position and direction.
 *
 * This is a helper class for \c HitProcessor.
 */
class NaviTouchableUpdater
{
  public:
    //! Maximum step to try within the current volume [len]
    static constexpr double max_step() { return 1 * units::millimeter; }

    //! Warn when the step is greater than this amount [len]
    static constexpr double max_quiet_step()
    {
        return 1e-3 * units::millimeter;
    }

    // Construct from touchable and navigator world
    explicit NaviTouchableUpdater(GeantTouchableBase* touchable);

    // Construct from touchable and explicit world
    NaviTouchableUpdater(GeantTouchableBase* touchable,
                         G4VPhysicalVolume const* world);

    // Default external deleter
    ~NaviTouchableUpdater();

    // Try to find the given point in the given logical volume
    bool
    operator()(Real3 const& pos, Real3 const& dir, G4LogicalVolume const* lv);

  private:
    std::unique_ptr<G4Navigator> navi_;
    GeantTouchableBase* touchable_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

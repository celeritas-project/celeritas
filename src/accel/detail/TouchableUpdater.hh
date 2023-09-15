//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/TouchableUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

class G4Navigator;
class G4VTouchable;
class G4LogicalVolume;

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
class TouchableUpdater
{
  public:
    //! Maximum step to try within the current volume [cm]
    static constexpr double max_step() { return 1 * units::millimeter; }

    //! Warn when the step is greater than this amount [cm]
    static constexpr double max_quiet_step()
    {
        return 1e-3 * units::millimeter;
    }

    // Construct with thread-local navigator and touchable
    inline TouchableUpdater(G4Navigator* navi, G4VTouchable* touchable);

    // Try to find the given point in the given logical volume
    bool
    operator()(Real3 const& pos, Real3 const& dir, G4LogicalVolume const* lv);

  private:
    G4Navigator* navi_;
    G4VTouchable* touchable_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with with thread-local navigator and touchable.
 */
TouchableUpdater::TouchableUpdater(G4Navigator* navi, G4VTouchable* touchable)
    : navi_{navi}, touchable_{touchable}
{
    CELER_EXPECT(navi_);
    CELER_EXPECT(touchable_);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

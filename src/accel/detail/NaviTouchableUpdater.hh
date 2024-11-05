//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/NaviTouchableUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "geocel/GeantGeoUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

class G4Navigator;
class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
struct DetectorStepOutput;

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
    //!@{
    //! \name Type aliases
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    //!@}

  public:
    //! Maximum step to try within the current volume [len]
    static constexpr double max_step() { return 1 * units::millimeter; }

    //! Warn when the step is greater than this amount [len]
    static constexpr double max_quiet_step()
    {
        return 1e-3 * units::millimeter;
    }

    // Construct from detector LVs
    explicit NaviTouchableUpdater(SPConstVecLV detector_volumes);

    // Construct from explicit world without detectors for unit testing
    explicit NaviTouchableUpdater(G4VPhysicalVolume const* world);

    // Construct from detector LVs and explicit world
    NaviTouchableUpdater(SPConstVecLV detector_volumes,
                         G4VPhysicalVolume const* world);

    // Default external deleter
    ~NaviTouchableUpdater();

    // Update from a particular detector step
    bool operator()(DetectorStepOutput const& out,
                    size_type step_index,
                    GeantTouchableBase* touchable);

    // Try to find the given point in the given logical volume
    bool operator()(Real3 const& pos,
                    Real3 const& dir,
                    G4LogicalVolume const* lv,
                    GeantTouchableBase* touchable);

  private:
    std::unique_ptr<G4Navigator> navi_;
    SPConstVecLV detector_volumes_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

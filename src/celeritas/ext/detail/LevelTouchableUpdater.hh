//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/LevelTouchableUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "geocel/GeantGeoUtils.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "TouchableUpdaterInterface.hh"

class G4Navigator;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4NavigationHistory;

namespace celeritas
{
//---------------------------------------------------------------------------//
struct DetectorStepOutput;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Update a Geant4 "touchable" using volume instances at each level.
 */
class LevelTouchableUpdater final : public TouchableUpdaterInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SpanVolInst = Span<VolumeInstanceId const>;
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    //!@}

  public:
    // Construct with the geometry
    explicit LevelTouchableUpdater(SPConstGeo);

    // Destroy pointers
    ~LevelTouchableUpdater() final;

    // Update from a particular detector step
    bool operator()(DetectorStepOutput const& out,
                    size_type step_index,
                    GeantTouchableBase* touchable) final;

    // Initialize from a span of volume instances
    bool operator()(SpanVolInst ids, GeantTouchableBase* touchable);

  private:
    // Geometry for doing G4PV translation
    SPConstGeo geo_;
    // Temporary storage for physical volumes
    std::vector<G4VPhysicalVolume const*> phys_vol_;
    // Temporary history
    std::unique_ptr<G4NavigationHistory> nav_hist_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

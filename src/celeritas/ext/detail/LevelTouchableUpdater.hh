//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/LevelTouchableUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "geocel/g4/GeantNavHistoryUpdater.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/user/DetectorSteps.hh"

#include "TouchableUpdaterInterface.hh"

class G4Navigator;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4NavigationHistory;

namespace celeritas
{
class GeantGeoParams;
//---------------------------------------------------------------------------//
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
    using SPConstGeantGeo = std::shared_ptr<GeantGeoParams const>;
    //!@}

  public:
    // Get a slice of volume instances from the output
    inline static SpanVolInst volume_instances(DetectorStepOutput const& out,
                                               size_type step_index,
                                               StepPoint step_point);

    // Construct with the geometry
    explicit LevelTouchableUpdater(SPConstGeantGeo);

    // Destroy pointers
    ~LevelTouchableUpdater() final;

    // Update from a particular detector step
    bool operator()(DetectorStepOutput const& out,
                    size_type step_index,
                    StepPoint step_point,
                    GeantTouchableBase* touchable) final;

    // Initialize from a span of volume instances
    bool operator()(SpanVolInst ids, GeantTouchableBase* touchable);

  private:
    SPConstGeantGeo geo_;
    // Temporary history
    std::unique_ptr<G4NavigationHistory> nav_hist_;
    // History updater
    GeantNavHistoryUpdater update_history_;
};

//---------------------------------------------------------------------------//
/*!
 * Get a slice of volume instances from the output.
 */
auto LevelTouchableUpdater::volume_instances(DetectorStepOutput const& out,
                                             size_type i,
                                             StepPoint sp) -> SpanVolInst
{
    CELER_EXPECT(i < out.size());
    CELER_EXPECT(out.volume_instance_depth > 0);
    CELER_EXPECT(!out.points[sp].volume_instance_ids.empty());
    auto ids = make_span(out.points[sp].volume_instance_ids);
    auto const depth = out.volume_instance_depth;
    CELER_EXPECT(ids.size() >= (i + 1) * depth);
    return ids.subspan(i * depth, depth);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

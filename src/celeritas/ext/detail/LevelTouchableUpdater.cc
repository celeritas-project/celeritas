//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/LevelTouchableUpdater.cc
//---------------------------------------------------------------------------//
#include "LevelTouchableUpdater.hh"

#include <G4NavigationHistory.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include "geocel/GeantGeoParams.hh"
#include "celeritas/user/DetectorSteps.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the geometry.
 */
LevelTouchableUpdater::LevelTouchableUpdater(SPConstGeantGeo geo)
    : geo_{std::move(geo)}
    , nav_hist_{std::make_unique<G4NavigationHistory>()}
    , update_history_{*geo_}
{
    CELER_EXPECT(geo_);
}

//---------------------------------------------------------------------------//
//! Destroy pointers.
LevelTouchableUpdater::~LevelTouchableUpdater() = default;

//---------------------------------------------------------------------------//
/*!
 * Update from a particular detector step.
 */
bool LevelTouchableUpdater::operator()(DetectorStepOutput const& out,
                                       size_type i,
                                       StepPoint step_point,
                                       GeantTouchableBase* touchable)
{
    return (*this)(LevelTouchableUpdater::volume_instances(out, i, step_point),
                   touchable);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize from a span of volume instances.
 *
 * Since the volume instances are allowed to be padded to better support GPU, a
 * null ID terminates the sequence. An empty input or one that starts with a
 * null ID indicates "outside".
 */
bool LevelTouchableUpdater::operator()(SpanVolInst ids,
                                       GeantTouchableBase* touchable)
{
    CELER_EXPECT(touchable);

    // Trim off trailing null volume instance IDs
    while (!ids.empty() && !ids.back())
    {
        ids = ids.subspan(0, ids.size() - 1);
    }

    // Use the volume instances to reconstruct a nav history
    update_history_(ids, nav_hist_.get());

    // Copy to the given touchable
    touchable->UpdateYourself(nav_hist_->GetTopVolume(), nav_hist_.get());
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

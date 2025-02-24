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

#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeoTraits.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/user/DetectorSteps.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the geometry.
 */
LevelTouchableUpdater::LevelTouchableUpdater(SPConstGeo geo)
    : geo_{std::move(geo)}, nav_hist_{std::make_unique<G4NavigationHistory>()}
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
                                       GeantTouchableBase* touchable)
{
    CELER_EXPECT(i < out.size());
    CELER_EXPECT(out.volume_instance_depth > 0);
    auto ids = make_span(out.points[StepPoint::pre].volume_instance_ids);
    auto const depth = out.volume_instance_depth;
    CELER_EXPECT(ids.size() >= (i + 1) * depth);

    return (*this)(ids.subspan(i * depth, depth), touchable);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize from a span of volume instances.
 */
bool LevelTouchableUpdater::operator()(SpanVolInst ids,
                                       GeantTouchableBase* touchable)
{
    CELER_EXPECT(touchable);
    CELER_EXPECT(!ids.empty() && ids.front());

    // Update phys_vol_ from geometry and volume instance id
    phys_vol_.clear();
    for (auto vi_id : ids)
    {
        if (!vi_id)
            break;
        auto pv = geo_->id_to_pv(vi_id);
        CELER_VALIDATE(
            pv,
            << "no Geant4 physical volume is attached to volume instance "
            << vi_id.get() << "='" << geo_->volume_instances().at(vi_id)
            << "' (geometry type: " << GeoTraits<GeoParams>::name << ')');
        phys_vol_.push_back(pv);
    }
    CELER_ASSERT(!phys_vol_.empty());

    set_history(make_span(phys_vol_), nav_hist_.get());
    touchable->UpdateYourself(nav_hist_->GetTopVolume(), nav_hist_.get());
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

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
#include "geocel/GeoParamsInterface.hh"
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

    // Update phys_inst_ from geometry and volume instance id
    phys_inst_.clear();
    for (auto vi_id : ids)
    {
        if (!vi_id)
            break;
        auto phys_inst = geo_->id_to_geant(vi_id);
        CELER_VALIDATE(
            phys_inst,
            << "no Geant4 physical volume is attached to volume instance "
            << vi_id.get() << "='" << geo_->volume_instances().at(vi_id)
            << "' (geometry type: " << GeoTraits<GeoParams>::name << ')');
        phys_inst_.push_back(phys_inst);
    }
    set_history(make_span(phys_inst_), nav_hist_.get());
    touchable->UpdateYourself(nav_hist_->GetTopVolume(), nav_hist_.get());
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/NaviTouchableUpdater.cc
//---------------------------------------------------------------------------//
#include "NaviTouchableUpdater.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Navigator.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/g4/Convert.hh"
#include "geocel/g4/Repr.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/user/DetectorSteps.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with world and volumes.
 */
NaviTouchableUpdater::NaviTouchableUpdater(SPConstVecLV detector_volumes,
                                           G4VPhysicalVolume const* world)
    : navi_{std::make_unique<G4Navigator>()}
    , detector_volumes_{std::move(detector_volumes)}
{
    CELER_EXPECT(world);
    CELER_EXPECT(detector_volumes_);

    navi_->SetWorldVolume(const_cast<G4VPhysicalVolume*>(world));

    CELER_ENSURE(navi_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with automatic navigation world.
 */
NaviTouchableUpdater::NaviTouchableUpdater(SPConstVecLV detector_volumes)
    : NaviTouchableUpdater{std::move(detector_volumes), geant_world_volume()}
{
}

//---------------------------------------------------------------------------//
//! Default external deleter
NaviTouchableUpdater::~NaviTouchableUpdater() = default;

//---------------------------------------------------------------------------//
/*!
 * Update from a particular detector step.
 */
bool NaviTouchableUpdater::operator()(DetectorStepOutput const& out,
                                      size_type i,
                                      StepPoint step_point,
                                      GeantTouchableBase* touchable)
{
    CELER_EXPECT(i < out.size());
    CELER_EXPECT(!out.points[StepPoint::pre].pos.empty());
    CELER_EXPECT(!out.points[StepPoint::pre].dir.empty());
    CELER_EXPECT(!out.detector.empty());
    CELER_EXPECT(detector_volumes_
                 && out.detector[i] < detector_volumes_->size());

    G4LogicalVolume const* lv
        = (*detector_volumes_)[out.detector[i].unchecked_get()];

    auto const& point = out.points[step_point];
    return (*this)(point.pos[i], point.dir[i], lv, touchable);
}

//---------------------------------------------------------------------------//
/*!
 * Try to find the given point in the given logical volume.
 *
 * Often on boundaries, the given logical volume (known from the VecGeom volume
 * mapping) is not consistent with the secondary Geant4 navigation volume.
 * If that's the case, try bumping forward and backward to see if we can enter
 * the correct volume.
 */
bool NaviTouchableUpdater::operator()(Real3 const& pos,
                                      Real3 const& dir,
                                      G4LogicalVolume const* lv,
                                      GeantTouchableBase* touchable)
{
    CELER_EXPECT(lv);
    CELER_EXPECT(touchable);

    auto g4pos = convert_to_geant(pos, clhep_length);
    auto g4dir = convert_to_geant(dir, 1);

    // Locate pre-step point
    navi_->LocateGlobalPointAndUpdateTouchable(g4pos,
                                               g4dir,
                                               touchable,
                                               /* relative_search = */ false);

    // Check whether physical and logical volumes are consistent
    G4VPhysicalVolume* pv = touchable->GetVolume(0);
    if (!pv)
    {
        // Exiting the world
        return true;
    }
    if (pv->GetLogicalVolume() == lv)
    {
        return true;
    }

    constexpr double g4max_step = convert_to_geant(max_step(), clhep_length);
    constexpr double g4max_quiet_step
        = convert_to_geant(max_quiet_step(), clhep_length);
    double g4safety{-1};
    double g4step{-1};

    //! Update next step and safety
    auto find_next_step = [&] {
        g4step = navi_->ComputeStep(g4pos, g4dir, g4max_step, g4safety);
    };

    //! Cross into the next touchable, updating the state and returning whether
    //! the volume is consistent.
    auto try_cross_boundary = [&] {
        if (g4step >= g4max_step)
        {
            // No nearby volumes in this direction
            return false;
        }
        else if (g4step > g4max_quiet_step)
        {
            // Warn since the step is nontrivial
            CELER_LOG_LOCAL(debug)
                << "Bumping navigation state by " << repr(g4step)
                << " [mm] at " << repr(g4pos) << " [mm] along " << repr(g4dir)
                << " from " << PrintableNavHistory{touchable->GetHistory()}
                << " to try to reach " << PrintableLV{lv};
        }

        // Move to boundary
        axpy(g4step, g4dir, &g4pos);
        navi_->SetGeometricallyLimitedStep();

        // Cross boundary
        navi_->LocateGlobalPointAndUpdateTouchable(
            g4pos,
            g4dir,
            touchable,
            /* relative_search = */ true);

        // Update volume and return whether it's correct
        pv = touchable->GetVolume(0);
        CELER_ASSERT(pv);

        if (g4step > g4max_quiet_step)
        {
            CELER_LOG_LOCAL(debug)
                << "Bumped navigation state by " << repr(g4step) << " to "
                << repr(g4pos) << " to enter "
                << PrintableNavHistory{touchable->GetHistory()};
        }

        return pv->GetLogicalVolume() == lv;
    };

    // First, find the next step along the current direction
    find_next_step();
    if (g4safety * 2 < g4step)
    {
        // Step forward is more than twice the known distance to boundary:
        // we're likely heading away from the closest intersection, so try that
        // first
        g4dir *= -1;
        find_next_step();
    }

    if (try_cross_boundary())
    {
        // Entered the correct volume
        return true;
    }

    // Reset the position and flip the direction
    g4pos = convert_to_geant(pos, clhep_length);
    g4dir *= -1;
    find_next_step();
    if (try_cross_boundary())
    {
        // Entered the correct volume
        return true;
    }

    // No nearby crossing found
    CELER_LOG_LOCAL(warning)
        << "Failed to bump navigation state up to a distance of " << g4max_step
        << " [mm] at " << repr(g4pos) << " [mm] along " << repr(g4dir)
        << " to try to reach " << PrintableLV{lv} << ": found "
        << PrintableNavHistory{touchable->GetHistory()};
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

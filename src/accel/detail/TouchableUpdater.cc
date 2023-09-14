//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/TouchableUpdater.cc
//---------------------------------------------------------------------------//
#include "TouchableUpdater.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Navigator.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/ext/Convert.geant.hh"
#include "celeritas/ext/GeantGeoParams.hh"
#include "celeritas/ext/GeantGeoUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<>
struct ReprTraits<G4ThreeVector>
{
    using value_type = std::decay_t<G4ThreeVector>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "G4ThreeVector";
        if (name)
        {
            os << ' ' << name;
        }
    }
    static void init(std::ostream& os) { ReprTraits<double>::init(os); }

    static void print_value(std::ostream& os, G4ThreeVector const& vec)
    {
        os << '{' << vec[0] << ", " << vec[1] << ", " << vec[2] << '}';
    }
};

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Try to find the given point in the given logical volume.
 *
 * Often on boundaries, the given logical volume (known from the VecGeom volume
 * mapping) is not consistent with the secondary Geant4 navigation volume.
 * If that's the case, try bumping forward and backward to see if we can enter
 * the correct volume.
 */
bool TouchableUpdater::operator()(Real3 const& pos,
                                  Real3 const& dir,
                                  G4LogicalVolume const* lv)
{
    auto g4pos = convert_to_geant(pos, CLHEP::cm);
    auto g4dir = convert_to_geant(dir, 1);

    // Locate pre-step point
    navi_->LocateGlobalPointAndUpdateTouchable(g4pos,
                                               g4dir,
                                               touchable_,
                                               /* relative_search = */ false);

    // Check whether physical and logical volumes are consistent
    G4VPhysicalVolume* pv = touchable_->GetVolume(0);
    CELER_ASSERT(pv);
    if (pv->GetLogicalVolume() == lv)
    {
        return true;
    }

    constexpr double g4max_step = convert_to_geant(max_step(), CLHEP::cm);
    constexpr double g4max_quiet_step
        = convert_to_geant(max_quiet_step(), CLHEP::cm);
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
            CELER_LOG_LOCAL(warning)
                << "Bumping navigation state by " << repr(g4step)
                << " [mm] at " << repr(g4pos) << " [mm] along " << repr(g4dir)
                << " from " << PrintableNavHistory{touchable_}
                << " to try to reach " << PrintableLV{lv};
        }

        // Move to boundary
        axpy(g4step, g4dir, &g4pos);
        navi_->SetGeometricallyLimitedStep();

        // Cross boundary
        navi_->LocateGlobalPointAndUpdateTouchable(
            g4pos,
            g4dir,
            touchable_,
            /* relative_search = */ true);

        // Update volume and return whether it's correct
        pv = touchable_->GetVolume(0);
        CELER_ASSERT(pv);

        if (g4step > g4max_quiet_step)
        {
            CELER_LOG_LOCAL(diagnostic)
                << "...bumped to " << PrintableNavHistory{touchable_};
        }
        else if (pv->GetLogicalVolume() == lv)
        {
            CELER_LOG_LOCAL(debug)
                << "Bumped navigation state by " << repr(g4step) << " to "
                << repr(g4pos) << " to enter "
                << PrintableNavHistory{touchable_};
        }

        return pv->GetLogicalVolume() == lv;
    };

    // First, find the next step along the current direction
    find_next_step();
    if (g4safety * 2 < g4step)
    {
        CELER_LOG_LOCAL(debug)
            << "Flipping search direction: safety " << g4safety
            << " [mm] is less than half of step " << g4step << " from "
            << PrintableLV{pv->GetLogicalVolume()} << " while trying to reach "
            << PrintableLV{lv};
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
    g4pos = convert_to_geant(pos, CLHEP::cm);
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
        << PrintableNavHistory{touchable_};
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

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

    // We may be accidentally in the old volume and crossing into
    // the new one: try crossing the edge. Use a fairly loose tolerance since
    // there may be small differences between the Geant4 and VecGeom
    // representations of the geometry.
    double safety_distance{-1};
    double step
        = navi_->ComputeStep(g4pos, g4dir, this->max_step(), safety_distance);
    if (step < this->max_step())
    {
        // Found a nearby volume
        if (step > this->max_quiet_step())
        {
            // Warn only if the step is nontrivial
            CELER_LOG_LOCAL(warning)
                << "Bumping navigation state by " << repr(step / CLHEP::mm)
                << " [mm] because the pre-step point at " << repr(g4pos)
                << " [mm] along " << repr(dir)
                << " is expected to be in logical volume " << PrintableLV{lv}
                << " but navigation gives " << PrintableNavHistory{touchable_};
        }

        navi_->SetGeometricallyLimitedStep();
        navi_->LocateGlobalPointAndUpdateTouchable(
            g4pos,
            g4dir,
            touchable_,
            /* relative_search = */ true);

        pv = touchable_->GetVolume(0);
        CELER_ASSERT(pv);
    }
    else
    {
        // No nearby crossing found
        CELER_LOG_LOCAL(warning)
            << "Failed to bump navigation state up to a distance of "
            << this->max_step() / CLHEP::mm << " [mm]";
    }

    return (pv->GetLogicalVolume() == lv);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

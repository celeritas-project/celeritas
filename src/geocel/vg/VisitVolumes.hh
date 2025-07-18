//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VisitVolumes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/VolumeVisitor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class VecgeomVolumeAccessor final
    : public VolumeAccessorInterface<vecgeom::LogicalVolume const*,
                                     vecgeom::VPlacedVolume const*>
{
  public:
    //! Outgoing volume node from an instance
    VolumeRef volume(VolumeInstanceRef parent) final
    {
        CELER_EXPECT(parent);
        auto result = parent->GetLogicalVolume();
        CELER_ENSURE(result);
        return result;
    }

    //! Outgoing instance nodes from a volume
    VecVolInstRef children(VolumeRef parent) final
    {
        auto&& daughters = parent->GetDaughters();
        return VecVolInstRef(daughters.begin(), daughters.end());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first traversal of physical volumes.
 *
 * The function must have the signature
 * <code>bool(*)(G4VPhysicalVolume const&, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, and the integer is the depth of the volume being visited.
 *
 * By default this will visit the entire "touchable" hierarchy: this may be
 * very expensive! If it's desired to only visit single physical volumes, mark
 * them as visited using a set.
 */
template<class F>
void visit_volume_instances(F&& vis, vecgeom::VPlacedVolume const* world)
{
    ScopedProfiling profile_this{"visit-vecgeom-volume-instance"};
    VolumeInstanceVisitor visit_vol{VecgeomVolumeAccessor{}};
    visit_vol(std::forward<F>(vis), world);
}

//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first listing of Geant4 logical volumes.
 *
 * This will visit each volume exactly once based on when it's encountered in
 * the hierarchy. The visitor function F should have the signature
 * \code void(*)(G4LogicalVolume const&) \endcode .
 */
template<class F>
void visit_volumes(F&& vis, vecgeom::VPlacedVolume const* world)
{
    ScopedProfiling profile_this{"visit-vecgeom-volume"};

    VolumeVisitor visit_vol{VecgeomVolumeAccessor{}};
    visit_vol(std::forward<F>(vis), world);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

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
    ContainerVolInstRef children(VolumeRef parent) final
    {
        auto&& daughters = parent->GetDaughters();
        return ContainerVolInstRef(daughters.begin(), daughters.end());
    }
};

//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first traversal of physical volumes.
 *
 * The function must have the signature
 * <code>bool(*)(vecgeom::VPlacedVolume const*, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, and the integer is the depth of the volume being visited.
 *
 * By default this will visit the entire "touchable" hierarchy: this may be
 * very expensive! If it's desired to only visit single physical volumes, mark
 * them as visited using a set.
 */
template<class Visitor>
void visit_volume_instances(Visitor&& v, vecgeom::VPlacedVolume const* world)
{
    ScopedProfiling profile_this{"visit-vecgeom-volume-instance"};
    VolumeVisitor visit_vol{VecgeomVolumeAccessor{}};
    visit_vol(std::forward<Visitor>(v), world);
}

//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first listing of Geant4 logical volumes.
 *
 * This will visit each volume exactly once based on when it's encountered in
 * the hierarchy. The visitor function Visitor should have the signature
 * \code void(*)(vecgeom::LogicalVolume const&) \endcode .
 */
template<class Visitor>
void visit_volumes(Visitor&& v, vecgeom::VPlacedVolume const* world)
{
    CELER_EXPECT(world);
    ScopedProfiling profile_this{"visit-vecgeom-volume"};

    VolumeVisitor visit_vol{VecgeomVolumeAccessor{}};
    visit_vol(make_visit_volume_once<vecgeom::LogicalVolume const*>(
                  std::forward<Visitor>(v)),
              world->GetLogicalVolume());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

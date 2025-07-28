//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/VisitVolumes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/VolumeVisitor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeantVolumeAccessor final
    : public VolumeAccessorInterface<G4LogicalVolume const*,
                                     G4VPhysicalVolume const*>
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

    //! Outgoing instance edges from a volume
    ContainerVolInstRef children(VolumeRef parent) final
    {
        CELER_EXPECT(parent);
        ContainerVolInstRef result;
        for (auto i : range(parent->GetNoDaughters()))
        {
            result.push_back(parent->GetDaughter(i));
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first traversal of physical volumes.
 *
 * The function must have the signature
 * <code>bool(*)(G4VPhysicalVolume const*, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, and the integer is the depth of the volume being visited.
 *
 * By default this will visit the entire "touchable" hierarchy: this may be
 * very expensive! If it's desired to only visit single physical volumes, mark
 * them as visited using a set.
 */
template<class Visitor>
void visit_volume_instances(Visitor&& vis, G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    ScopedProfiling profile_this{"visit-geant-volume-instance"};
    VolumeVisitor visit_vol{GeantVolumeAccessor{}};
    visit_vol(std::forward<Visitor>(vis), world);
}

//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first traversal of Geant4 logical volumes.
 *
 * This will visit each volume exactly once based on when it's encountered in
 * the hierarchy. The visitor function Visitor should have the signature
 * \code void(*)(G4LogicalVolume const*) \endcode .
 */
template<class Visitor>
void visit_volumes(Visitor&& vis, G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    ScopedProfiling profile_this{"visit-geant-volume"};

    VolumeVisitor visit_vol{GeantVolumeAccessor{}};
    visit_vol(make_visit_volume_once<G4LogicalVolume const*>(
                  std::forward<Visitor>(vis)),
              world->GetLogicalVolume());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

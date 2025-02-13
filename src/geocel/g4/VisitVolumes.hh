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
template<>
struct VolumeVisitorTraits<G4VPhysicalVolume>
{
    using PV = G4VPhysicalVolume;
    using LV = G4LogicalVolume;

    static LV const& get_lv(PV const& pv) { return *pv.GetLogicalVolume(); }

    static void get_children(PV const& parent, std::vector<PV const*>& dst)
    {
        LV const& lv = get_lv(parent);
        auto num_children = lv.GetNoDaughters();
        for (auto i : range(num_children))
        {
            dst.push_back(lv.GetDaughter(i));
        }
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
void visit_volume_instances(F&& visit, G4VPhysicalVolume const& world)
{
    ScopedProfiling profile_this{"visit-geant-volume-instance"};
    VolumeVisitor{world}(std::forward<F>(visit));
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
void visit_volumes(F&& vis, G4VPhysicalVolume const& parent_vol)
{
    ScopedProfiling profile_this{"visit-geant-volume"};

    visit_logical_volumes(std::forward<F>(vis), parent_vol);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

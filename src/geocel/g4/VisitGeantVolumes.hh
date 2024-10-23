//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/VisitGeantVolumes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_set>
#include <vector>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform a depth-first traversal of physical volumes.
 *
 * The function must have the signature
 * <code>bool(*)(G4VPhysicalVolume const&, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, and the integer is the depth of the volume being visited.
 *
 * By default this will visit the entire "touchable" hierachy: this may be very
 * expensive! If it's desired to only visit single physical volumes, mark them
 * as visited using a set.
 */
template<class F>
void visit_geant_volume_instances(F&& visit, G4VPhysicalVolume const& world)
{
    struct QueuedVolume
    {
        G4VPhysicalVolume const* pv{nullptr};
        int depth{0};
    };

    std::vector<QueuedVolume> queue;
    auto visit_impl = [&queue, &visit](G4VPhysicalVolume const& pv, int depth) {
        if (visit(pv, depth))
        {
            // Append children
            auto const* lv = pv.GetLogicalVolume();
            CELER_ASSERT(lv);
            auto num_children = lv->GetNoDaughters();
            for (auto i : range(num_children))
            {
                queue.push_back(
                    {lv->GetDaughter(num_children - 1 - i), depth + 1});
            }
        }
    };

    // Visit the top-level physical volume
    visit_impl(world, 0);

    while (!queue.empty())
    {
        QueuedVolume qv = queue.back();
        queue.pop_back();

        // Visit popped daughter
        visit_impl(*qv.pv, qv.depth);
    }
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
void visit_geant_volumes(F&& vis, G4VPhysicalVolume const& parent_vol)
{
    std::unordered_set<G4LogicalVolume const*> visited;
    auto visit_impl
        = [&vis, &visited](G4VPhysicalVolume const& pv, int) -> bool {
        auto const* lv = pv.GetLogicalVolume();
        if (!visited.insert(lv).second)
        {
            // Already visited
            return false;
        }
        vis(*lv);
        return true;
    };

    visit_geant_volume_instances(visit_impl, parent_vol);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

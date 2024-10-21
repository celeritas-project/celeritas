//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/VisitGeantVolumes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <deque>
#include <unordered_set>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform a breadth-first traversal of physical volumes.
 *
 * The function must have the signature
 * <code>bool(*)(G4VPhysicalVolume const&, int)</code>
 * where the return value indicates whether the volume's daughters should be
 * visited, and the integer is the depth of the volume being visited.
 */
template<class F>
void visit_geant_volume_instances(F&& visit, G4VPhysicalVolume const& world)
{
    struct QueuedDaughter
    {
        G4VPhysicalVolume const* pv{nullptr};
        int depth{0};
    };

    std::deque<QueuedDaughter> queue;
    auto visit_impl
        = [&queue, &visit](G4VPhysicalVolume const& g4pv, int depth) {
              if (visit(g4pv, depth))
              {
                  // Append children
                  auto const* lv = g4pv.GetLogicalVolume();
                  CELER_ASSERT(lv);
                  auto num_children = lv->GetNoDaughters();
                  for (auto i : range(num_children))
                  {
                      queue.push_back({lv->GetDaughter(i), depth + 1});
                  }
              }
          };

    // Visit the top-level physical volume
    visit_impl(world, 0);

    while (!queue.empty())
    {
        QueuedDaughter qd = queue.front();
        queue.pop_front();

        // Visit popped daughter
        visit_impl(*qd.pv, qd.depth);
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

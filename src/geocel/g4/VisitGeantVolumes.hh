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

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Do a recursive depth-first listing of Geant4 logical volumes.
 *
 * This will visit each volume exactly once based on when it's encountered in
 * the hierarchy. The visitor function F should have the signature
 * \code void(*)(G4LogicalVolume const&) \endcode .
 */
template<class F>
void visit_geant_volumes(F&& vis, G4LogicalVolume const& parent_vol)
{
    std::unordered_set<G4LogicalVolume const*> visited;
    std::vector<G4LogicalVolume const*> stack{&parent_vol};

    while (!stack.empty())
    {
        G4LogicalVolume const* lv = stack.back();
        stack.pop_back();
        vis(*lv);
        for (auto const i : range(lv->GetNoDaughters()))
        {
            G4LogicalVolume* daughter = lv->GetDaughter(i)->GetLogicalVolume();
            CELER_ASSERT(daughter);
            auto&& [iter, inserted] = visited.insert(daughter);
            if (inserted)
            {
                stack.push_back(daughter);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

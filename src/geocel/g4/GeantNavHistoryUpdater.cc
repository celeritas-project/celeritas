//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantNavHistoryUpdater.cc
//---------------------------------------------------------------------------//
#include "GeantNavHistoryUpdater.hh"

#include <G4NavigationHistory.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/math/Algorithms.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/detail/GeantVolumeInstanceMapper.hh"

using G4PV = G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct using geant geo params.
 */
GeantNavHistoryUpdater::GeantNavHistoryUpdater(GeantGeoParams const& geo)
    : GeantNavHistoryUpdater{*geo.host_ref().vi_mapper}
{
}

//---------------------------------------------------------------------------//
/*!
 * Update a nav history to match the given volume instance stack.
 */
void GeantNavHistoryUpdater::operator()(Span<VolumeInstanceId const> stack,
                                        G4NavigationHistory* nav)
{
    CELER_EXPECT(std::all_of(stack.begin(), stack.end(), Identity{}));
    CELER_EXPECT(nav);

    VolumeLevelId level{0};
    auto nav_stack_size
        = [nav] { return static_cast<size_type>(nav->GetDepth()) + 1; };

    // Loop deeper until stack and nav disagree. Use the replica number stored
    // in the history rather than the PV's copy number: the latter is shared
    // state that may have been changed by another navigator since this
    // level was added.
    for (auto end_level = id_cast<VolumeLevelId>(
             std::min<size_type>(stack.size(), nav_stack_size()));
         level != end_level;
         ++level)
    {
        if (mapper_.geant_to_id(*nav, level) != stack[*level])
        {
            break;
        }
    }

    if (CELER_UNLIKELY(level == VolumeLevelId{0}))
    {
        // Top level disagrees: this should likely only happen when we're
        // outside (i.e. stack is empty)
        nav->Reset();
        if (!stack.empty())
        {
            auto& pv = mapper_.id_to_geant(stack[0]);
            nav->SetFirstEntry(const_cast<G4PV*>(&pv));
            ++level;
        }
        else
        {
            nav->SetFirstEntry(nullptr);
        }
    }
    else if (level < nav_stack_size())
    {
        // Decrease nav stack to the parent's level
        nav->BackLevel(nav_stack_size() - *level);
        CELER_ASSERT(nav_stack_size() == *level);
    }

    // Add all remaining levels: see G4Navigator::LocateGlobalPoint
    // Note that the mapper's id_to_geant functionality updates the G4PV
    // appropriately if a replica
    for (auto end_level = id_cast<VolumeLevelId>(stack.size());
         level != end_level;
         ++level)
    {
        auto& pv = mapper_.id_to_geant(stack[*level]);
        nav->NewLevel(const_cast<G4PV*>(&pv), pv.VolumeType(), pv.GetCopyNo());
    }

    CELER_ENSURE(nav_stack_size() == stack.size()
                 || (stack.empty() && nav->GetDepth() == 0));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

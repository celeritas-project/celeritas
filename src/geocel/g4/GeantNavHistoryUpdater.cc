//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantNavHistoryUpdater.cc
//---------------------------------------------------------------------------//
#include "GeantNavHistoryUpdater.hh"

#include <G4NavigationHistory.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/OpaqueId.hh"
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

    auto const init_nav_levels = detail::get_num_volume_levels(*nav);

    // Loop deeper until stack and nav disagree (note that the nav encodes the
    // replica ID for parameterised volumes)
    VolumeLevelId level{0};
    auto end_level
        = std::min(init_nav_levels, id_cast<VolumeLevelId>(stack.size()));
    while (level != end_level
           && (mapper_.geant_to_id(*nav, level) == stack[*level]))
    {
        ++level;
    }

    if (CELER_UNLIKELY(level == VolumeLevelId{0}))
    {
        // Top level disagrees: this should likely only happen when we're
        // outside (i.e. stack is empty) or given an empty navigator
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
    else if (level < init_nav_levels)
    {
        // Decrease nav stack to the parent's level
        nav->BackLevel(init_nav_levels - level);
        CELER_ASSERT(detail::get_num_volume_levels(*nav) == level);
    }

    // Add all remaining levels: see G4Navigator::LocateGlobalPoint
    // Note that the mapper's id_to_geant functionality updates the G4PV
    // appropriately if a replica
    for (end_level = id_cast<VolumeLevelId>(stack.size()); level != end_level;
         ++level)
    {
        auto& pv = mapper_.id_to_geant(stack[*level]);
        nav->NewLevel(const_cast<G4PV*>(&pv), pv.VolumeType(), pv.GetCopyNo());
    }

    // Depth will match unless it's empty
    CELER_ENSURE(level == id_cast<VolumeLevelId>(stack.size()));
    CELER_ENSURE(detail::get_num_volume_levels(*nav) == level);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

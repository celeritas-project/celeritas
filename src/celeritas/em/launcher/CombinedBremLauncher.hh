//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "sim/CoreTrackView.hh"

#include "CombinedBremData.hh"
#include "CombinedBremInteractor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply electron/positron brems to the current track.
 */
inline CELER_FUNCTION Interaction combined_brem_interact_track(
    CombinedBremRef const& model, CoreTrackView const& track)
{
    // Select material track view
    auto material = track.make_material_view().make_material_view();

    // Assume only a single element in the material, for now
    CELER_ASSERT(material.num_elements() == 1);
    const ElementComponentId selected_element{0};

    auto        particle             = track.make_particle_view();
    const auto& dir                  = track.make_geo_view().dir();
    auto        allocate_secondaries = track.make_secondary_allocator();
    auto        cutoff               = track.make_cutoff_view();

    CombinedBremInteractor interact(model,
                                    particle,
                                    dir,
                                    cutoff,
                                    allocate_secondaries,
                                    material,
                                    selected_element);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

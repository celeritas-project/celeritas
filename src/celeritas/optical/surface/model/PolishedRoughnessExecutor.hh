//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Trivial executor for the polished roughness model.
 *
 * Sets the facet normal of the track to the global normal of the track.
 */
struct PolishedRoughnessExecutor
{
    //! Apply polished roughness executor
    CELER_FUNCTION void operator()(CoreTrackView& track) const
    {
        auto surface_phys = track.surface_physics();
        surface_phys.facet_normal(surface_phys.global_normal());
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

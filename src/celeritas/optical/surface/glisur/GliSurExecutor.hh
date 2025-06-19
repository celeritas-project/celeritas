//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct GliSurExecutor
{
    NativeCRef<GliSurData> data;

    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

CELER_FUNCTION void GliSurExecutor::operator()(CoreTrackView const& track) const
{
    auto surface = track.surface();

    // Select facet normal action based on surface finish
    if (data.finish[surface.surface_id()] == GliSurFinishType::ground)
    {
        surface.set_normal_action(data.scalars.glisur_polished_normal_action);
    }
    else
    {
        surface.set_normal_action(data.scalars.trivial_normal_action);
    }

    // Always use grid-based reflectivity calculations
    surface.set_calc_reflectivity_action(data.scalars.grid_reflectivity_action);

    // Select interaction based on interface type
    if (data.interface_type[surface.surface_id()]
        == GliSurInterfaceType::dielectric_metal)
    {
        surface.set_interaction_action(data.scalars.glisur_metal_interaction);
    }
    else
    {
        surface.set_interaction_action(
            data.scalars.glisur_dielectric_interaction);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

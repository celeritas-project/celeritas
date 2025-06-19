//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurPolishedNormalExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct GliSurPolishedNormalExecutor
{
    NativeCRef<GliSurPolishedNormalData> data;

    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

CELER_FUNCTION void
GliSurPolishedNormalExecutor::operator()(CoreTrackView const& track) const
{
    auto surface = track.surface();

    // auto model_id = surface.model_id();
    // auto model_surface_id = surface.model_surface_id();
    // auto polish_table = data.polish_table[model_id];
    // auto polish_id = polish_table[model_surface_id];
    // real_type polish = data.polish[polish_id];

    real_type polish
        = data.polish[data.polish_table[surface.model_id()]
                                       [surface.model_surface_id()]]

          auto geo
        = track.geometry();
    auto rng = track.rng();

    GliSurPolishedNormalCalculator sample_normal{
        surface.surface_normal(), polish, geo.dir()};
    Real3 facet_normal = sample_normal(rng);

    CELER_ASSERT(is_soft_unit_vector(facet_normal));
    CELER_ASSERT(dot_product(facet_normal, geo.dir()) < 0);

    track.set_normal(facet_normal);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

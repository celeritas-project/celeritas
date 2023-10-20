//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/RelativisticBremExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/interactor/RelativisticBremInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RelativisticBremExecutor
{
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 900
    inline static constexpr int max_block_size{320};
#elif CELERITAS_USE_CUDA
    inline static constexpr int max_block_size{CELERITAS_MAX_BLOCK_SIZE};
#endif
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    RelativisticBremRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply RelativisticBrem to the current track.
 */
CELER_FUNCTION Interaction
RelativisticBremExecutor::operator()(CoreTrackView const& track)
{
    auto cutoff = track.make_cutoff_view();
    auto material = track.make_material_view().make_material_view();
    auto particle = track.make_particle_view();

    auto elcomp_id = track.make_physics_step_view().element();
    CELER_ASSERT(elcomp_id);
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto const& dir = track.make_geo_view().dir();

    RelativisticBremInteractor interact(
        params, particle, dir, cutoff, allocate_secondaries, material, elcomp_id);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

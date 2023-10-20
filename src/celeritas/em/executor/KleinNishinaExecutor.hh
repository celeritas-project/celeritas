//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/KleinNishinaExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/em/interactor/KleinNishinaInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct KleinNishinaExecutor
{
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 900
    inline static constexpr int max_block_size{192};
#elif CELERITAS_USE_CUDA
    inline static constexpr int max_block_size{CELERITAS_MAX_BLOCK_SIZE};
#endif
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    KleinNishinaData params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the KleinNishinaInteractor to the current track.
 */
CELER_FUNCTION Interaction
KleinNishinaExecutor::operator()(CoreTrackView const& track)
{
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto particle = track.make_particle_view();
    auto const& dir = track.make_geo_view().dir();

    KleinNishinaInteractor interact(
        params, particle, dir, allocate_secondaries);
    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

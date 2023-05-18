//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitUtils.hh
//! \brief Helper functions for initializing tracks
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "TrackInitData.hh"
#include "generated/ProcessPrimaries.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from a vector of host primary particles.
 */
template<MemSpace M>
inline void extend_from_primaries(CoreParams const& core_params,
                                  CoreState<M>& core_state,
                                  Span<Primary const> host_primaries)
{
    CELER_EXPECT(!host_primaries.empty());

    auto& init = core_state.ref().init;
    CELER_ASSERT(host_primaries.size() + init.scalars.num_initializers
                 <= init.initializers.size());

    // Resizing the initializers is a non-const operation, but the only one.
    init.scalars.num_initializers += host_primaries.size();

    // Allocate memory and copy primaries
    Collection<Primary, Ownership::value, M> primaries;
    resize(&primaries, host_primaries.size());
    Copier<Primary, M> copy_to_temp{primaries[AllItems<Primary, M>{}]};
    copy_to_temp(MemSpace::host, host_primaries);

    // Create track initializers from primaries
    generated::process_primaries(core_params.ref<M>(),
                                 core_state.ref(),
                                 primaries[AllItems<Primary, M>{}]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

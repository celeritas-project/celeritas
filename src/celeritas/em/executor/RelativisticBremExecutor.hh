//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    auto cutoff = track.cutoff();
    auto material = track.material().material_record();
    auto particle = track.particle();

    auto elcomp_id = track.physics_step().element();
    CELER_ASSERT(elcomp_id);
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto const& dir = track.geometry().dir();

    RelativisticBremInteractor interact(
        params, particle, dir, cutoff, allocate_secondaries, material, elcomp_id);

    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

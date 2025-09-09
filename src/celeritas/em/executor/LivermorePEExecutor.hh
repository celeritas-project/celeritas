//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/LivermorePEExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/interactor/LivermorePEInteractor.hh"
#include "celeritas/em/xs/LivermorePEMicroXsCalculator.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/random/ElementSelector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct LivermorePEExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<LivermorePEData> params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample a Livermore photoelectric interaction from the current track.
 */
CELER_FUNCTION Interaction
LivermorePEExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto rng = track.rng();

    // Get the element ID if an element was previously sampled
    auto elcomp_id = track.physics_step().element();
    if (!elcomp_id)
    {
        // Sample an element (calculating microscopic cross sections on the
        // fly) and store it
        auto material = track.material();
        auto material_record = material.material_record();
        ElementSelector select_el(
            material_record,
            LivermorePEMicroXsCalculator{params, particle.energy()},
            material.element_scratch());
        elcomp_id = select_el(rng);
        CELER_ASSERT(elcomp_id);
        track.physics_step().element(elcomp_id);
    }
    auto el_id = track.material().material_record().element_id(elcomp_id);

    // Set up photoelectric interactor with the selected element
    auto relaxation = track.physics_step().make_relaxation_helper(el_id);
    auto cutoffs = track.cutoff();
    auto const& dir = track.geometry().dir();
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    LivermorePEInteractor interact(
        params, relaxation, el_id, particle, cutoffs, dir, allocate_secondaries);

    // Sample the interaction
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

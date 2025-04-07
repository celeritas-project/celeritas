//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/CombinedBremExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/CombinedBremData.hh"
#include "celeritas/em/interactor/CombinedBremInteractor.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CombinedBremExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    CombinedBremRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample electron/positron brems from the current track.
 */
CELER_FUNCTION Interaction
CombinedBremExecutor::operator()(CoreTrackView const& track)
{
    // Select material track view
    auto material = track.material().material_record();

    // Assume only a single element in the material, for now
    CELER_ASSERT(material.num_elements() == 1);
    ElementComponentId const selected_element{0};

    auto particle = track.particle();
    auto const& dir = track.geometry().dir();
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto cutoff = track.cutoff();

    CombinedBremInteractor interact(params,
                                    particle,
                                    dir,
                                    cutoff,
                                    allocate_secondaries,
                                    material,
                                    selected_element);

    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

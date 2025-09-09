//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SlotDiagnosticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../ParticleTallyData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Export the particle type.
 *
 * Later this class can be extended to write different properties per track,
 * e.g. action ID; or save into the "track" slot versus
 *
 * \note
 */
struct SlotDiagnosticExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    ObserverPtr<int> output;
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void
SlotDiagnosticExecutor::operator()(CoreTrackView const& track)
{
    int result = [&track] {
        auto sim = track.sim();
        if (sim.status() == TrackStatus::inactive)
        {
            return -1;
        }
        else if (sim.status() == TrackStatus::errored)
        {
            return -2;
        }
        // Save particle ID
        return static_cast<int>(track.particle().particle_id().get());
    }();

    size_type const index = track.track_slot_id().get();
    (&*output)[index] = result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

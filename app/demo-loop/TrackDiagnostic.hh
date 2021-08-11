//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Diagnostic.hh"

#include "base/Macros.hh"
#include "physics/base/ModelInterface.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackInterface.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Example diagnostic class for collecting track data on device.
 *
 * This trivial example collects the number of surviving tracks after a
 * given simulation step through a reduction on a state's tracks.
 *
 */

template<MemSpace M>
struct SumAlive
{
    using result_type = size_type;

    StateData<Ownership::reference, MemSpace::device> states;

    CELER_FUNCTION size_type operator()(size_type num_alive, ThreadId tid) const
    {
        if (tid.get() >= states.size())
            return num_alive;

        SimTrackView sim(states.sim, tid);
        if (!sim.alive())
            return num_alive;

        return num_alive + sim.alive() ? 1 : 0;
    }

    // Symmetric operator
    CELER_FUNCTION size_type operator()(ThreadId tid, size_type num_alive) const
    {
        return (*this)(num_alive, tid);
    }
};

template<MemSpace M>
class TrackDiagnostic : public Diagnostic<M>
{
  public:
    using StateDataDeviceRef = StateData<Ownership::reference, M>;

    TrackDiagnostic() : Diagnostic<M>() {}

    void end_step(const StateDataDeviceRef& data) final;

    inline std::vector<size_type> num_alive_per_step()
    {
        return num_alive_per_step_;
    }

  private:
    std::vector<size_type> num_alive_per_step_;
};

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
size_type
reduce_alive(const StateData<Ownership::reference, MemSpace::device>& states);
} // namespace demo_loop
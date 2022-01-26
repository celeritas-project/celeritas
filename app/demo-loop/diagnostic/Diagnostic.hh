//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Diagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "physics/base/ModelData.hh"
#include "sim/TrackData.hh"
#include "../Transporter.hh"

using celeritas::MemSpace;
using celeritas::Ownership;

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Interface for on-device data access at different stages of a simulation.
 */
template<MemSpace M>
class Diagnostic
{
  public:
    using EventId           = celeritas::EventId;
    using StateDataRef      = celeritas::StateData<Ownership::reference, M>;
    using TransporterResult = celeritas::TransporterResult;

    // Virtual destructor for polymorphic deletion
    virtual ~Diagnostic() = 0;

    // Memory allocations
    virtual void begin_simulation() {}

    // Collect diagnostic(s) before event begins
    virtual void begin_event(EventId, const StateDataRef&) {}

    // Collect diagnostic(s) before step
    virtual void begin_step(const StateDataRef&) {}

    // Collect diagnostic(s) in the middle of a step
    virtual void mid_step(const StateDataRef&) {}

    // Collect diagnostic(s) after step
    virtual void end_step(const StateDataRef&) {}

    // Collect diagnostic(s) after event ends
    virtual void end_event(EventId, const StateDataRef&) {}

    // Collect post-sim diagnostic(s)
    virtual void end_simulation() {}

    // Collect results from diagnostic
    virtual void get_result(TransporterResult*) {}
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Default destructor
template<MemSpace M>
inline Diagnostic<M>::~Diagnostic()
{
}

//---------------------------------------------------------------------------//
} // namespace demo_loop

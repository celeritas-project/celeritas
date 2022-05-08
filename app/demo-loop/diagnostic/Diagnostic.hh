//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/diagnostic/Diagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"

using celeritas::MemSpace;
using celeritas::Ownership;

namespace celeritas
{
struct TransporterResult;
}

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
    using EventId  = celeritas::EventId;
    using StateRef = celeritas::CoreStateData<Ownership::reference, M>;
    using TransporterResult = celeritas::TransporterResult;

    // Virtual destructor for polymorphic deletion
    virtual ~Diagnostic() = 0;

    // Collect diagnostic(s) in the middle of a step
    virtual void mid_step(const StateRef&) {}

    // Collect diagnostic(s) after step
    virtual void end_step(const StateRef&) {}

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

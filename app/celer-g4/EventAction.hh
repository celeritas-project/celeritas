//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/EventAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserEventAction.hh>

#include "corecel/sys/Stopwatch.hh"
#include "accel/LocalTransporter.hh"
#include "accel/SharedParams.hh"

#include "GeantDiagnostics.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Manage begin- and end-of-event setup.
 *
 * This class is local to a thread/task/stream, but it shares \c
 * LocalTransporter with other user actions on the current thread.
 */
class EventAction final : public G4UserEventAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<SharedParams const>;
    using SPTransporter = std::shared_ptr<LocalTransporter>;
    using SPDiagnostics = std::shared_ptr<GeantDiagnostics>;
    //!@}

  public:
    EventAction(SPConstParams params,
                SPTransporter transport,
                SPDiagnostics diagnostics);

    void BeginOfEventAction(G4Event const* event) final;
    void EndOfEventAction(G4Event const* event) final;

  private:
    SPConstParams params_;
    SPTransporter transport_;
    SPDiagnostics diagnostics_;
    Stopwatch get_event_time_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

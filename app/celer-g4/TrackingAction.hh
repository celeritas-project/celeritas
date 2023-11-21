//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TrackingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserTrackingAction.hh>

#include "accel/LocalTransporter.hh"
#include "accel/SharedParams.hh"

#include "GeantDiagnostics.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Offload EM tracks to Celeritas.
 *
 * This class is local to a thread/task/stream. It shares \c SharedParams with
 * all threads/tasks, and it shares \c LocalTransporter with other user actions
 * on the current thread.
 */
class TrackingAction final : public G4UserTrackingAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<SharedParams const>;
    using SPDiagnostics = std::shared_ptr<GeantDiagnostics>;
    using SPTransporter = std::shared_ptr<LocalTransporter>;
    //!@}

  public:
    TrackingAction(SPConstParams params,
                   SPTransporter transport,
                   SPDiagnostics diagnostics);

    void PreUserTrackingAction(G4Track const* track) final;
    void PostUserTrackingAction(G4Track const* track) final;

  private:
    SPConstParams params_;
    SPTransporter transport_;
    SPDiagnostics diagnostics_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

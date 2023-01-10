//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/TrackingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserTrackingAction.hh>

#include "accel/LocalTransporter.hh"
#include "accel/SharedParams.hh"

namespace demo_geant
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
    using SPConstParams = std::shared_ptr<const celeritas::SharedParams>;
    using SPTransporter = std::shared_ptr<celeritas::LocalTransporter>;
    //!@}

  public:
    TrackingAction(SPConstParams params, SPTransporter transport);

    void PreUserTrackingAction(const G4Track* track) final;

  private:
    SPConstParams params_;
    SPTransporter transport_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant

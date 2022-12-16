//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4UserTrackingAction.hh>

#include "SharedParams.hh"
#include "detail/LocalTransporter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Offload EM tracks to Celeritas.
 */
class TrackingAction final : public G4UserTrackingAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<const SharedParams>;
    using SPTransporter = std::shared_ptr<detail::LocalTransporter>;
    //!@}

  public:
    TrackingAction(SPConstParams params, SPTransporter transport);

    void PreUserTrackingAction(const G4Track* track) final;

  private:
    SPConstParams params_;
    SPTransporter transport_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

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
 *  EM tracks to the device.
 */
class TrackingAction final : public G4UserTrackingAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams      = std::shared_ptr<SharedParams>;
    using SPTransporter = std::shared_ptr<detail::LocalTransporter>;
    //!@}

  public:
    TrackingAction(SPParams params, SPTransporter transport);

    void PreUserTrackingAction(const G4Track* track) final;

  private:
    SPParams      params_;
    SPTransporter transport_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

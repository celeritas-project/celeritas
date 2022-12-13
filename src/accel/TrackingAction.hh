//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4UserTrackingAction.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Offload EM tracks to Celeritas.
 */
class TrackingAction final : public G4UserTrackingAction
{
  public:
    TrackingAction();

    void PreUserTrackingAction(const G4Track* track) final;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

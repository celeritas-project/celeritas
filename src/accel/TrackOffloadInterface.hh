//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackOffloadInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "LocalOffloadInterface.hh"

class G4Track;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interface for offloading complete Geant4 tracks to Celeritas.
 *
 * It allows the Geant4 tracking manager to forward full
 * track to Celeritas, such as EM or optical track transport.
 */
class TrackOffloadInterface : public LocalOffloadInterface
{
  public:
    // Construct with defaults
    ~TrackOffloadInterface() override = default;

    // Push a full Geant4 track to Celeritas
    virtual void Push(G4Track&) = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

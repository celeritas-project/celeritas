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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    TrackOffloadInterface ...;
   \endcode
 */
class TrackOffloadInterface : public LocalOffloadInterface
{
  public:
    // Construct with defaults
    virtual ~TrackOffloadInterface() = default;

    // Push a full Geant4 track to Celeritas
    virtual void Push(G4Track&) = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "base/Types.hh"
#include "SimStatePointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage on-device simulation states.
 */
class SimStateStore
{
  public:
    // Construct from number of track states
    explicit SimStateStore(size_type size);

    /// ACCESSORS ///

    // Number of states
    size_type size() const;

    // View on-device states
    SimStatePointers device_pointers();

  private:
    DeviceVector<SimTrackState> vars_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

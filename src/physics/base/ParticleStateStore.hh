//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "base/Types.hh"
#include "ParticleStatePointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage on-device particle physics states.
 */
class ParticleStateStore
{
  public:
    // Construct from number of track states
    explicit ParticleStateStore(size_type size);

    // >>> ACCESSORS

    // Number of states
    size_type size() const;

    // View on-device states
    ParticleStatePointers device_pointers();

  private:
    DeviceVector<ParticleTrackState> vars_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//

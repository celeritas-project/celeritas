//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Primary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "ParticleInterface.hh"
#include "sim/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Starting "source" particle. One or more of these are emitted by an Event.
 */
struct Primary
{
    ParticleId       particle_id;
    units::MevEnergy energy;
    Real3            position;
    Real3            direction;
    EventId          event_id;
    TrackId          track_id;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

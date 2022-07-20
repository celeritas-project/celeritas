//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Primary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Types.hh"

#include "ParticleData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Starting "source" particle.
 *
 * One or more of these are emitted by an Event.
 */
struct Primary
{
    ParticleId       particle_id;
    units::MevEnergy energy;
    Real3            position;
    Real3            direction;
    real_type        time;
    EventId          event_id;
    TrackId          track_id;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

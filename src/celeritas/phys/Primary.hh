//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Primary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Starting "source" particle.
 *
 * One or more of these are emitted by an Event.
 *
 * The event ID is the local event slot used for counting tracks. When
 * reseeding each track, the RNG is seeded from the primary ID and the unique
 * event ID, or the local event ID if the unique one is not set.
 */
struct Primary
{
    ParticleId particle_id;
    units::MevEnergy energy;
    Real3 position{0, 0, 0};
    Real3 direction{0, 0, 0};
    real_type time{};
    real_type weight{1.0};

    EventId event_id;
    PrimaryId primary_id;
    UniqueEventId unique_event_id;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

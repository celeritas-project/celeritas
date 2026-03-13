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
 */
struct Primary
{
    ParticleId particle_id;
    units::MevEnergy energy;
    Real3 position{0, 0, 0};
    Real3 direction{0, 0, 0};
    real_type time{};
    real_type weight{1.0};

    // Primary and event IDs are used for RNG seeding
    EventId event_id;
    PrimaryId primary_id;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

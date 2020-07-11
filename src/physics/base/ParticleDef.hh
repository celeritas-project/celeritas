//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleDef.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"
#include "base/Types.hh"

namespace celeritas
{
struct ParticleDef;
using ParticleDefId = OpaqueId<ParticleDef>;

//---------------------------------------------------------------------------//
/*!
 * Fundamental (static) properties of a particle type.
 *
 * These should only be fundamental physical properties. Setting particles is
 * done through the ParticleParams. Physical state of a particle
 * (kinetic energy, ...) is part of a ParticleState.
 *
 * Particle definitions are accessed via the ParticleParams: using PDGs
 * to look up particle IDs, etc.
 */
struct ParticleDef
{
    real_type mass;           // Rest mass [MeV / c]
    real_type charge;         // Charge in units of [e]
    real_type decay_constant; // Decay constant [1/s]

    //! Value of decay_constant for a stable particle
    static CELER_CONSTEXPR_FUNCTION real_type stable_decay_constant()
    {
        return 0;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//

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
#include "Units.hh"

namespace celeritas
{
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
    units::MevMass          mass;           //!< Rest mass [MeV / c^2]
    units::ElementaryCharge charge;         //!< Charge in units of [e]
    real_type               decay_constant; //!< Decay constant [1/s]

    //! Value of decay_constant for a stable particle
    static CELER_CONSTEXPR_FUNCTION real_type stable_decay_constant()
    {
        return 0;
    }
};

//! Opaque index to ParticleDef in a vector: represents a particle type
using ParticleDefId = OpaqueId<ParticleDef>;

//---------------------------------------------------------------------------//
} // namespace celeritas

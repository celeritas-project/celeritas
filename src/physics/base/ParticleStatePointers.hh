//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleStatePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "ParticleDef.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physical (dynamic) state of a particle track.
 *
 * The "physical state" is just about what differentiates this particle from
 * another (type, energy, polarization, ...) in the lab's inertial reference
 * frame. It does not include information about the particle's direction or
 * position, nor about path lengths or collisions.
 *
 * The energy is with respect to the lab frame. The particle state is
 * immutable: collisions and other interactions should return changes to the
 * particle state.
 */
struct ParticleTrackState
{
    ParticleDefId particle_type;  //!< Type of particle (electron, gamma, ...)
    real_type     kinetic_energy; //!< Kinetic energy [MeV]
};

//---------------------------------------------------------------------------//
/*!
 * View to the dynamic states of multiple physical particles.
 */
struct ParticleStatePointers
{
    span<ParticleTrackState> vars;

    //! Check whether the view is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !vars.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//

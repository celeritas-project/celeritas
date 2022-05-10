//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ParticleData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant particle data.
 *
 * A \c ParticleView can be used to access properties of a single particle
 * type, e.g. electron or photon.
 */
class ParticleView
{
  public:
    //!@{
    //! Type aliases
    using ParticleParamsRef
        = ParticleParamsData<Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from "static" particle definitions
    inline CELER_FUNCTION ParticleView(const ParticleParamsRef&, ParticleId);

    // Unique particle type identifier
    CELER_FORCEINLINE_FUNCTION ParticleId particle_id() const;

    // Rest mass [MeV / c^2]
    CELER_FORCEINLINE_FUNCTION units::MevMass mass() const;

    // Charge [elemental charge e+]
    CELER_FORCEINLINE_FUNCTION units::ElementaryCharge charge() const;

    // Decay constant [1/s]
    CELER_FORCEINLINE_FUNCTION real_type decay_constant() const;

  private:
    const ParticleParamsRef& params_;
    const ParticleId         particle_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from "static" particle definitions.
 */
CELER_FUNCTION
ParticleView::ParticleView(const ParticleParamsRef& params, ParticleId id)
    : params_(params), particle_(id)
{
    CELER_EXPECT(particle_ < params_.particles.size());
}

//---------------------------------------------------------------------------//
/*!
 * Unique particle type identifier.
 */
CELER_FUNCTION ParticleId ParticleView::particle_id() const
{
    return particle_;
}

//---------------------------------------------------------------------------//
/*!
 * Rest mass [MeV / c^2].
 */
CELER_FUNCTION units::MevMass ParticleView::mass() const
{
    return params_.particles[particle_].mass;
}

//---------------------------------------------------------------------------//
/*!
 * Elementary charge.
 */
CELER_FUNCTION units::ElementaryCharge ParticleView::charge() const
{
    return params_.particles[particle_].charge;
}

//---------------------------------------------------------------------------//
/*!
 * Decay constant.
 */
CELER_FUNCTION real_type ParticleView::decay_constant() const
{
    return params_.particles[particle_].decay_constant;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

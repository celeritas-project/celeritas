//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
    //! \name Type aliases
    using ParticleParamsRef = NativeCRef<ParticleParamsData>;
    //!@}

  public:
    // Construct from "static" particle definitions
    inline CELER_FUNCTION ParticleView(ParticleParamsRef const&, ParticleId);

    // Unique particle type identifier
    CELER_FORCEINLINE_FUNCTION ParticleId particle_id() const;

    // Rest mass [MeV / c^2]
    CELER_FORCEINLINE_FUNCTION units::MevMass mass() const;

    // Charge [elemental charge e+]
    CELER_FORCEINLINE_FUNCTION units::ElementaryCharge charge() const;

    // Decay constant [1/s]
    CELER_FORCEINLINE_FUNCTION real_type decay_constant() const;

    // Whether it is an antiparticle
    CELER_FORCEINLINE_FUNCTION bool is_antiparticle() const;

  private:
    ParticleParamsRef const& params_;
    const ParticleId particle_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from "static" particle definitions.
 */
CELER_FUNCTION
ParticleView::ParticleView(ParticleParamsRef const& params, ParticleId id)
    : params_(params), particle_(id)
{
    CELER_EXPECT(particle_ < params_.size());
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
    return params_.mass[particle_];
}

//---------------------------------------------------------------------------//
/*!
 * Elementary charge.
 */
CELER_FUNCTION units::ElementaryCharge ParticleView::charge() const
{
    return params_.charge[particle_];
}

//---------------------------------------------------------------------------//
/*!
 * Decay constant.
 */
CELER_FUNCTION real_type ParticleView::decay_constant() const
{
    return params_.decay_constant[particle_];
}

//---------------------------------------------------------------------------//
/*!
 * Whether it is an antiparticle.
 */
CELER_FUNCTION bool ParticleView::is_antiparticle() const
{
    return params_.matter[particle_] == MatterType::antiparticle;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

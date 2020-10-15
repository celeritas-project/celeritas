//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "ParticleStatePointers.hh"
#include "ParticleParamsPointers.hh"
#include "Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read/write view to the physical properties of a single particle track.
 *
 * These functions should be used in each physics Process or Interactor or
 * anything else that needs to access particle properties. Assume that all
 * these functions are expensive: when using them as accessors, locally store
 * the results rather than calling the function repeatedly. If any of the
 * calculations prove to be hot spots we will experiment with cacheing some of
 * the variables.
 */
class ParticleTrackView
{
  public:
    //@{
    //! Type aliases
    using Initializer_t = ParticleTrackState;
    //@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION
    ParticleTrackView(const ParticleParamsPointers& params,
                      const ParticleStatePointers&  states,
                      ThreadId                      id);

    // Initialize the particle
    inline CELER_FUNCTION ParticleTrackView&
                          operator=(const Initializer_t& other);

    // Change the particle's energy [MeV]
    inline CELER_FUNCTION void energy(units::MevEnergy);

    // >>> DYNAMIC PROPERTIES (pure accessors, free)

    // Unique particle type identifier
    inline CELER_FUNCTION ParticleDefId def_id() const;

    // Kinetic energy [MeV]
    inline CELER_FUNCTION units::MevEnergy energy() const;

    // >>> STATIC PROPERTIES (requires an indirection)

    // Rest mass [MeV / c^2]
    inline CELER_FUNCTION units::MevMass mass() const;

    // Charge [elemental charge e+]
    inline CELER_FUNCTION units::ElementaryCharge charge() const;

    // Decay constant [1/s]
    inline CELER_FUNCTION real_type decay_constant() const;

    // >>> DERIVED PROPERTIES (indirection plus calculation)

    // Speed [1/c]
    inline CELER_FUNCTION units::LightSpeed speed() const;

    // Lorentz factor [unitless]
    inline CELER_FUNCTION real_type lorentz_factor() const;

    // Relativistic momentum [MeV / c]
    inline CELER_FUNCTION units::MevMomentum momentum() const;

    // Relativistic momentum squared [MeV^2 / c^2]
    inline CELER_FUNCTION units::MevMomentumSq momentum_sq() const;

  private:
    const ParticleParamsPointers& params_;
    ParticleTrackState&           state_;

    inline CELER_FUNCTION const ParticleDef& particle_def() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ParticleTrackView.i.hh"

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
#include "ParticleInterface.hh"
#include "ParticleView.hh"
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
    //!@{
    //! Type aliases
    using ParticleParamsRef
        = ParticleParamsData<Ownership::const_reference, MemSpace::native>;
    using ParticleStateRef
        = ParticleStateData<Ownership::reference, MemSpace::native>;
    using Initializer_t = ParticleTrackState;
    //!@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION ParticleTrackView(const ParticleParamsRef& params,
                                            const ParticleStateRef&  states,
                                            ThreadId                 id);

    // Initialize the particle
    inline CELER_FUNCTION ParticleTrackView&
                          operator=(const Initializer_t& other);

    // Change the particle's energy [MeV]
    inline CELER_FUNCTION void energy(units::MevEnergy);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Unique particle type identifier
    CELER_FORCEINLINE_FUNCTION ParticleId particle_id() const;

    // Kinetic energy [MeV]
    CELER_FORCEINLINE_FUNCTION units::MevEnergy energy() const;

    // Whether the particle is stopped (zero kinetic energy)
    CELER_FORCEINLINE_FUNCTION bool is_stopped() const;

    //// STATIC PROPERTIES (requires an indirection) ////

    CELER_FORCEINLINE_FUNCTION ParticleView particle_view() const;

    // Rest mass [MeV / c^2]
    CELER_FORCEINLINE_FUNCTION units::MevMass mass() const;

    // Charge [elemental charge e+]
    CELER_FORCEINLINE_FUNCTION units::ElementaryCharge charge() const;

    // Decay constant [1/s]
    CELER_FORCEINLINE_FUNCTION real_type decay_constant() const;

    //// DERIVED PROPERTIES (indirection plus calculation) ////

    // Speed [1/c]
    inline CELER_FUNCTION units::LightSpeed speed() const;

    // Lorentz factor [unitless]
    inline CELER_FUNCTION real_type lorentz_factor() const;

    // Relativistic momentum [MeV / c]
    inline CELER_FUNCTION units::MevMomentum momentum() const;

    // Relativistic momentum squared [MeV^2 / c^2]
    inline CELER_FUNCTION units::MevMomentumSq momentum_sq() const;

  private:
    const ParticleParamsRef& params_;
    const ParticleStateRef&  states_;
    const ThreadId           thread_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ParticleTrackView.i.hh"

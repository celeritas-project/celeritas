//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"

#include "ParticleData.hh"
#include "ParticleView.hh"

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
    //! \name Type aliases
    using ParticleParamsRef = celeritas::NativeCRef<ParticleParamsData>;
    using ParticleStateRef = celeritas::NativeRef<ParticleStateData>;
    using Energy = units::MevEnergy;
    using Initializer_t = ParticleTrackInitializer;
    //!@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION ParticleTrackView(ParticleParamsRef const& params,
                                            ParticleStateRef const& states,
                                            TrackSlotId id);

    // Initialize the particle
    inline CELER_FUNCTION ParticleTrackView&
    operator=(Initializer_t const& other);

    // Change the particle's energy [MeV]
    inline CELER_FUNCTION void energy(Energy);

    // Reduce the particle's energy [MeV]
    inline CELER_FUNCTION void subtract_energy(Energy);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Unique particle type identifier
    CELER_FORCEINLINE_FUNCTION ParticleId particle_id() const;

    // Kinetic energy [MeV]
    CELER_FORCEINLINE_FUNCTION Energy energy() const;

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

    // Whether it is an antiparticle
    CELER_FORCEINLINE_FUNCTION bool is_antiparticle() const;

    // Whether the particle is stable
    CELER_FORCEINLINE_FUNCTION bool is_stable() const;

    //// DERIVED PROPERTIES (indirection plus calculation) ////

    // Square of fraction of lightspeed [unitless]
    inline CELER_FUNCTION real_type beta_sq() const;

    // Speed [1/c]
    inline CELER_FUNCTION units::LightSpeed speed() const;

    // Lorentz factor [unitless]
    inline CELER_FUNCTION real_type lorentz_factor() const;

    // Relativistic momentum [MeV / c]
    inline CELER_FUNCTION units::MevMomentum momentum() const;

    // Relativistic momentum squared [MeV^2 / c^2]
    inline CELER_FUNCTION units::MevMomentumSq momentum_sq() const;

  private:
    ParticleParamsRef const& params_;
    ParticleStateRef const& states_;
    const TrackSlotId track_slot_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
ParticleTrackView::ParticleTrackView(ParticleParamsRef const& params,
                                     ParticleStateRef const& states,
                                     TrackSlotId tid)
    : params_(params), states_(states), track_slot_(tid)
{
    CELER_EXPECT(track_slot_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the particle.
 */
CELER_FUNCTION ParticleTrackView&
ParticleTrackView::operator=(Initializer_t const& other)
{
    CELER_EXPECT(other.particle_id < params_.particles.size());
    CELER_EXPECT(other.energy >= zero_quantity());
    states_.state[track_slot_].particle_id = other.particle_id;
    states_.state[track_slot_].energy = other.energy.value();
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Change the particle's kinetic energy.
 *
 * This should only be used when the particle is in a valid state. For HEP
 * applications, the new energy should always be less than the starting energy.
 */
CELER_FUNCTION
void ParticleTrackView::energy(Energy quantity)
{
    CELER_EXPECT(this->particle_id());
    CELER_EXPECT(quantity >= zero_quantity());
    states_.state[track_slot_].energy = quantity.value();
}

//---------------------------------------------------------------------------//
/*!
 * Reduce the particle's energy without undergoing a collision [MeV].
 */
CELER_FUNCTION void ParticleTrackView::subtract_energy(Energy eloss)
{
    CELER_EXPECT(eloss >= zero_quantity());
    CELER_EXPECT(eloss <= this->energy());
    // TODO: save a read/write by only saving if eloss is positive?
    states_.state[track_slot_].energy -= eloss.value();
}

//---------------------------------------------------------------------------//
// DYNAMIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Unique particle type identifier.
 */
CELER_FUNCTION ParticleId ParticleTrackView::particle_id() const
{
    return states_.state[track_slot_].particle_id;
}

//---------------------------------------------------------------------------//
/*!
 * Kinetic energy [MeV].
 */
CELER_FUNCTION auto ParticleTrackView::energy() const -> Energy
{
    return Energy{states_.state[track_slot_].energy};
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is stopped (zero kinetic energy).
 */
CELER_FUNCTION bool ParticleTrackView::is_stopped() const
{
    return this->energy() == zero_quantity();
}

//---------------------------------------------------------------------------//
// STATIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Get static particle properties for the current state.
 */
CELER_FUNCTION ParticleView ParticleTrackView::particle_view() const
{
    return ParticleView(params_, states_.state[track_slot_].particle_id);
}

//---------------------------------------------------------------------------//
/*!
 * Rest mass [MeV / c^2].
 */
CELER_FUNCTION units::MevMass ParticleTrackView::mass() const
{
    return this->particle_view().mass();
}

//---------------------------------------------------------------------------//
/*!
 * Elementary charge.
 */
CELER_FUNCTION units::ElementaryCharge ParticleTrackView::charge() const
{
    return this->particle_view().charge();
}

//---------------------------------------------------------------------------//
/*!
 * Decay constant.
 */
CELER_FUNCTION real_type ParticleTrackView::decay_constant() const
{
    return this->particle_view().decay_constant();
}

//---------------------------------------------------------------------------//
/*!
 * Whether it is an antiparticle.
 */
CELER_FUNCTION bool ParticleTrackView::is_antiparticle() const
{
    return this->particle_view().is_antiparticle();
}

//---------------------------------------------------------------------------//
/*!
 * Whether particle is stable.
 */
CELER_FUNCTION bool ParticleTrackView::is_stable() const
{
    return this->decay_constant() == ParticleRecord::stable_decay_constant();
}

//---------------------------------------------------------------------------//
// COMBINED PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Square of \f$ \beta \f$, which is the fraction of lightspeed [unitless].
 *
 * Beta is calculated using the equality
 * \f[
 * pc/E = \beta ; \quad \beta^2 = \frac{p^2 c^2}{E^2}
 * \f].
 * Using
 * \f[
 * E^2 = p^2 c^2 + m^2 c^4
 * \f]
 * and
 * \f[
 * E = K + mc^2
 * \f]
 *
 * the square of the fraction of lightspeed speed is
 * \f[
 * \beta^2 = 1 - \left( \frac{mc^2}{K + mc^2} \right)^2
 *         = 1 - \gamma^{-2}
 * \f]
 *
 * where \f$ \gamma \f$ is the Lorentz factor (see below).
 *
 * By choosing not to divide out the mass, this expression will work for
 * massless particles.
 *
 * \todo For nonrelativistic particles (low kinetic energy) this expression
 * may be inaccurate due to rounding error. It is however guaranteed to be in
 * the exact range [0, 1].
 */
CELER_FUNCTION real_type ParticleTrackView::beta_sq() const
{
    // Rest mass as energy
    const real_type mcsq = this->mass().value();
    // Inverse of lorentz factor (safe for m=0)
    real_type inv_gamma = mcsq / (this->energy().value() + mcsq);

    return 1 - ipow<2>(inv_gamma);
}

//---------------------------------------------------------------------------//
/*!
 * Speed [1/c].
 *
 * Speed is calculated using beta so that the expression works for massless
 * particles.
 */
CELER_FUNCTION units::LightSpeed ParticleTrackView::speed() const
{
    return units::LightSpeed{std::sqrt(this->beta_sq())};
}

//---------------------------------------------------------------------------//
/*!
 * Lorentz factor [unitless].
 *
 * The Lorentz factor can be viewed as a transformation from
 * classical quantities to relativistic quantities. It's defined as
 * \f[
  \gamma = \frac{1}{\sqrt{1 - v^2 / c^2}}
  \f]
 *
 * Its value is infinite for the massless photon, and greater than or equal to
 * 1 otherwise.
 *
 * Gamma can also be calculated from the total (rest + kinetic) energy
 * \f[
  E = \gamma mc^2 = K + mc^2
  \f]
 * which we ues here since \em K and \em m are the primary stored quantities of
 * the particles:
 * \f[
  \gamma = 1 + \frac{K}{mc^2}
 * \f]
 */
CELER_FUNCTION real_type ParticleTrackView::lorentz_factor() const
{
    CELER_EXPECT(this->mass() > zero_quantity());

    real_type k_over_mc2 = this->energy().value() / this->mass().value();
    return 1 + k_over_mc2;
}

//---------------------------------------------------------------------------//
/*!
 * Square of relativistic momentum [MeV^2 / c^2].
 *
 * Total energy:
 * \f[
 * E = K + mc^2
 * \f]
 * Relation between energy and momentum:
 * \f[
 * E^2 = p^2 c^2 + m^2 c^4
 * \f]
 * therefore
 * \f[
 * p^2 = \frac{E^2}{c^2} - m^2 c^2
 * \f]
 * or
 * \f[
 * p^2 = \frac{K^2}{c^2} + 2 * m * K
 * \f]
 */
CELER_FUNCTION units::MevMomentumSq ParticleTrackView::momentum_sq() const
{
    const real_type energy = this->energy().value();
    real_type result = ipow<2>(energy) + 2 * this->mass().value() * energy;
    CELER_ENSURE(result >= 0);
    return units::MevMomentumSq{result};
}

//---------------------------------------------------------------------------//
/*!
 * Relativistic momentum [MeV / c].
 *
 * This is calculated by taking the root of the square of the momentum.
 */
CELER_FUNCTION units::MevMomentum ParticleTrackView::momentum() const
{
    return units::MevMomentum{std::sqrt(this->momentum_sq().value())};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

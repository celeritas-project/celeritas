//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleTrackView.i.hh
//---------------------------------------------------------------------------//

#include <cmath>

#include "base/Assert.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
ParticleTrackView::ParticleTrackView(const ParticleParamsPointers& params,
                                     const ParticleStatePointers&  states,
                                     ThreadId                      id)
    : params_(params), state_(states.vars[id.get()])
{
    REQUIRE(id < states.vars.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the particle.
 */
CELER_FUNCTION ParticleTrackView&
ParticleTrackView::operator=(const Initializer_t& other)
{
    REQUIRE(other.def_id < params_.defs.size());
    REQUIRE(other.energy >= zero_quantity());
    state_ = other;
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
void ParticleTrackView::energy(units::MevEnergy quantity)
{
    REQUIRE(this->def_id());
    REQUIRE(quantity >= zero_quantity());
    state_.energy = quantity;
}

//---------------------------------------------------------------------------//
// DYNAMIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Unique particle type identifier.
 */
CELER_FUNCTION ParticleDefId ParticleTrackView::def_id() const
{
    return state_.def_id;
}

//---------------------------------------------------------------------------//
/*!
 * Kinetic energy [MeV].
 */
CELER_FUNCTION units::MevEnergy ParticleTrackView::energy() const
{
    return state_.energy;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is stopped (zero kinetic energy).
 */
CELER_FUNCTION bool ParticleTrackView::is_stopped() const
{
    return state_.energy == zero_quantity();
}

//---------------------------------------------------------------------------//
// STATIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Rest mass [MeV / c^2].
 */
CELER_FUNCTION units::MevMass ParticleTrackView::mass() const
{
    return this->particle_def().mass;
}

//---------------------------------------------------------------------------//
/*!
 * Elementary charge.
 */
CELER_FUNCTION units::ElementaryCharge ParticleTrackView::charge() const
{
    return this->particle_def().charge;
}

//---------------------------------------------------------------------------//
/*!
 * Decay constant.
 */
CELER_FUNCTION real_type ParticleTrackView::decay_constant() const
{
    return this->particle_def().decay_constant;
}

//---------------------------------------------------------------------------//
/*!
 * Speed [1/c].
 *
 * Speed is calculated using the equality pc/E = v/c --> v = pc^2/E. Using
 * \f[
 * E^2 = p^2 c^2 + m^2 c^4
 * \f]
 * and
 * \f[
 * E = K + mc^2
 * \f]
 *
 * the speed can be simplified to
 * \f[
 * v = c \sqrt{1 - (mc^2 / (T + mc^2))^2} = c \sqrt{1 - \gamma^{-2}}
 * \f]
 *
 * where \f$ \gamma \f$ is the Lorentz factor (see below).
 *
 * By choosing not to divide out the mass, this expression will work for
 * massless particles.
 */
CELER_FUNCTION units::LightSpeed ParticleTrackView::speed() const
{
    // Rest mass as energy
    real_type mcsq = this->mass().value();
    // Inverse of lorentz factor (safe for m=0)
    real_type inv_gamma = mcsq / (this->energy().value() + mcsq);

    return units::LightSpeed{std::sqrt(1 - inv_gamma * inv_gamma)};
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
    REQUIRE(this->mass() > zero_quantity());

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
 * p = \frac{K^2}{c^2} + 2 * m * K
 * \f]
 */
CELER_FUNCTION units::MevMomentumSq ParticleTrackView::momentum_sq() const
{
    const real_type energy = this->energy().value();
    real_type result = energy * energy + 2 * this->mass().value() * energy;
    ENSURE(result > 0);
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
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Get static particle defs for the current state.
 */
CELER_FUNCTION const ParticleDef& ParticleTrackView::particle_def() const
{
    REQUIRE(state_.def_id < params_.defs.size());
    return params_.defs[state_.def_id.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas

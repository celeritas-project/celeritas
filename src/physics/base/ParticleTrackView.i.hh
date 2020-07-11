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
 * \brief Initialize the particle
 */
CELER_FUNCTION ParticleTrackView&
ParticleTrackView::operator=(const Initializer_t& other)
{
    REQUIRE(other.particle_type < params_.defs.size());
    REQUIRE(other.kinetic_energy > 0);
    state_ = other;
    return *this;
}

//---------------------------------------------------------------------------//
// DYNAMIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Unique particle type identifier.
 */
CELER_FUNCTION ParticleDefId ParticleTrackView::particle_type() const
{
    return state_.particle_type;
}

//---------------------------------------------------------------------------//
/*!
 * Kinetic energy [MeV].
 */
CELER_FUNCTION real_type ParticleTrackView::kinetic_energy() const
{
    return state_.kinetic_energy;
}

//---------------------------------------------------------------------------//
// STATIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Rest mass [MeV / c^2].
 */
CELER_FUNCTION real_type ParticleTrackView::mass() const
{
    return this->particle_def().mass;
}

//---------------------------------------------------------------------------//
/*!
 * Elementary charge.
 */
CELER_FUNCTION real_type ParticleTrackView::charge() const
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
 * Speed is calculated using the equality pc/E = v/c --> v = pc^2/E.
 * This expression can be used to both massive and massless particles. When
 * m = 0, p = E/c, which yields v = c.
 */
CELER_FUNCTION real_type ParticleTrackView::speed() const
{
    // Calculating total energy: E = sqrt(p^c^2 + m^2c^4)
    real_type c2 = units::speed_of_light_sq;
    real_type c4 = c2 * c2;
    real_type p2 = this->momentum() * this->momentum();
    real_type m2 = this->mass() * this->mass();

    real_type E = std::sqrt(p2 * c2 + m2 * c4);

    // Returning v = pc^2/E
    return this->momentum() * c2 / E;
}

//---------------------------------------------------------------------------//
/*!
 * Lorentz factor [unitless].
 *
 * Massless particles: Cannot be calculated (gamma = infinity)
 * Massive particles : E = gamma.mc^2 = K + mc^2 --> gamma = 1 + K/(mc^2)
 */
CELER_FUNCTION real_type ParticleTrackView::lorentz_factor() const
{
    REQUIRE(this->mass() > 0);

    real_type k_over_mc2 = this->kinetic_energy()
                           / (this->mass() * units::speed_of_light_sq);
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
 * E^2 = p^2 c^2 + m^2 c^4 \to
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
CELER_FUNCTION real_type ParticleTrackView::momentum_sq() const
{
    constexpr real_type inv_c_sq = 1 / units::speed_of_light_sq;

    const real_type kinetic_energy = this->kinetic_energy();
    real_type       result         = kinetic_energy * kinetic_energy * inv_c_sq
                       + 2 * this->mass() * kinetic_energy;
    ENSURE(result > 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Relativistic momentum [MeV / c].
 *
 * Total energy:
 * \f[
 * E = K + mc^2
 * \f]
 * Relation between energy and momentum:
 * \f[
 * E^2 = p^2 c^2 + m^2 c^4 \to
 * \f]
 * therefore
 * \f[
 * p = \sqrt{\frac{E^2}{c^2} - m^2 c^2}
 * \f]
 * or
 * \f[
 * p = \sqrt{\frac{K^2}{c^2} + 2 * m * K}
 * \f]
 */
CELER_FUNCTION real_type ParticleTrackView::momentum() const
{
    return std::sqrt(this->momentum_sq());
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Get static particle defs for the current state.
 */
CELER_FUNCTION const ParticleDef& ParticleTrackView::particle_def() const
{
    REQUIRE(state_.particle_type < params_.defs.size());
    return params_.defs[state_.particle_type.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas

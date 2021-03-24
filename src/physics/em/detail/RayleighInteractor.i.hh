//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "random/distributions/IsotropicDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
RayleighInteractor::RayleighInteractor(const RayleighNativePointers& shared,
                                       const ParticleTrackView&      particle,
                                       const Real3&       inc_direction,
                                       const ElementView& element)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , element_(element)
{
    CELER_EXPECT(inc_energy_ >= this->min_incident_energy()
                 && inc_energy_ <= this->max_incident_energy());
    CELER_EXPECT(particle.particle_id() == shared_.gamma_id);
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the G4LivermoreRayleighModel model.
 */
template<class Engine>
CELER_FUNCTION Interaction RayleighInteractor::operator()(Engine& rng)
{
    real_type energy = inc_energy_.value();

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy = units::MevEnergy{energy};

    // Sample direction for a given Z: G4RayleighAngularGenerator
    unsigned int Z = element_.atomic_number();
    using ItemIdT  = celeritas::ItemId<int>;
    Array<real_type, rayleigh_num_parameters> params
        = shared_.params.data[ItemIdT{Z - 1}];

    real_type xx = form_factor() * form_factor() * energy * energy;

    real_type n0 = params[6] - 1.0;
    real_type n1 = params[7] - 1.0;
    real_type n2 = params[8] - 1.0;
    real_type b0 = params[3];
    real_type b1 = params[4];
    real_type b2 = params[5];

    real_type w0 = this->evaluate_weight(xx * b0, n0);
    real_type w1 = this->evaluate_weight(xx * b1, n1);
    real_type w2 = this->evaluate_weight(xx * b2, n2);

    real_type x0 = w0 * params[0] / (b0 * n0);
    real_type x1 = w1 * params[1] / (b1 * n1);
    real_type x2 = w2 * params[2] / (b2 * n2);

    real_type                          cost;
    UniformRealDistribution<real_type> u01(0, 1.0);

    do
    {
        real_type w = w0;
        real_type n = n0;
        real_type b = b0;
        real_type x = u01(rng) * (x0 + x1 + x2);

        if (x > x0)
        {
            x -= x0;
            if (x <= x1)
            {
                w = w1;
                n = n1;
                b = b1;
            }
            else
            {
                w = w2;
                n = n2;
                b = b2;
            }
        }
        n = 1.0 / n;

        // Sampling of scattering angle
        real_type y = w * u01(rng);
        if (y < num_limit())
        {
            x = y * n * (1. + 0.5 * (n + 1.) * y * (1. - (n + 2.) * y / 3.));
        }
        else
        {
            x = std::exp(-n * std::log(1. - y)) - 1.0;
        }

        cost = 1.0 - 2.0 * x / (b * xx);

    } while (2 * u01(rng) > 1.0 + cost * cost || cost < -1.0);

    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Scattered direction
    result.direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    return result;
}

CELER_FUNCTION
real_type RayleighInteractor::evaluate_weight(real_type x, real_type nx) const
{
    return (x > num_limit())
               ? 1.0 - std::exp(-nx * std::log(1.0 + x))
               : nx * x
                     * (1.0
                        - 0.5 * (nx - 1.0) * x * (1.0 - (nx - 2.0) * x / 3.));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

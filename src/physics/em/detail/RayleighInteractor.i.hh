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
                                       const Real3&                  direction,
                                       ElementId                     el_id)

    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(direction)
    , element_id_(el_id)
{
    CELER_EXPECT(inc_energy_ >= this->min_incident_energy()
                 && inc_energy_ <= this->max_incident_energy());
    CELER_EXPECT(particle.particle_id() == shared_.gamma_id);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the Rayleigh scattering angle using the G4LivermoreRayleighModel
 * and G4RayleighAngularGenerator of Geant4 6.10.
 */
template<class Engine>
CELER_FUNCTION Interaction RayleighInteractor::operator()(Engine& rng)
{
    real_type energy = inc_energy_.value();

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy = units::MevEnergy{inc_energy_.value()};

    // Sample direction for a given Z: G4RayleighAngularGenerator
    ItemIdT item_id = ItemIdT{element_id_.get()};

    Real3 pn = shared_.params.data_n[item_id];
    Real3 pb = shared_.params.data_b[item_id];
    Real3 px = shared_.params.data_x[item_id];

    SampleInput input = this->evaluate_weight_and_prob(energy, pb, pn, px);

    real_type                          cost;
    UniformRealDistribution<real_type> u01(0, 1.0);

    do
    {
        unsigned  index = 0;
        real_type x     = u01(rng);

        if (x > input.prob[0])
        {
            index = (x <= input.prob[0] + input.prob[1]) ? 1 : 2;
        }

        real_type w = input.weight[index];
        real_type n = pn[index];
        real_type b = pb[index];

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

        cost = 1.0 - 2.0 * x / (b * input.factor);

    } while (2 * u01(rng) > 1.0 + cost * cost || cost < -1.0);

    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Scattered direction
    result.direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    return result;
}

CELER_FUNCTION
auto RayleighInteractor::evaluate_weight_and_prob(real_type    energy,
                                                  const Real3& b,
                                                  const Real3& n,
                                                  const Real3& px) const
    -> SampleInput
{
    SampleInput input;

    real_type factor = energy * hc_factor();
    input.factor     = factor * factor;

    Real3 x = b;
    axpy(input.factor, b, &x);

    Real3 prob;
    for (auto i : range(3))
    {
        input.weight[i]
            = (x[i] > num_limit())
                  ? 1.0 - std::exp(-n[i] * std::log(1.0 + x[i]))
                  : n[i] * x[i]
                        * (1.0
                           - 0.5 * (n[i] - 1.0) * (x[i])
                                 * (1.0 - (n[i] - 2.0) * (x[i]) / 3.));

        prob[i] = input.weight[i] * px[i] / (b[i] * n[i]);
    }

    real_type inv_sum = 1.0 / (prob[0] + prob[1] + prob[2]);
    axpy(inv_sum, prob, &input.prob);

    return input;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

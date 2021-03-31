//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"
#include "random/distributions/GenerateCanonical.hh"
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
RayleighInteractor::RayleighInteractor(const RayleighNativeRef& shared,
                                       const ParticleTrackView& particle,
                                       const Real3&             direction,
                                       ElementId                el_id)

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

    // Sample direction for a given atomic number: G4RayleighAngularGenerator
    ItemIdT item_id = ItemIdT{element_id_.get()};

    SampleInput input = this->evaluate_weight_and_prob(energy, item_id);

    Real3 pb = shared_.params.data[item_id].b;
    Real3 pn = shared_.params.data[item_id].n;

    constexpr real_type half = 0.5;
    real_type           cost;

    do
    {
        unsigned int index = 0;
        // Sample index from input.prob
        {
            real_type u = generate_canonical(rng);
            if (u > input.prob[0])
            {
                index = (u <= input.prob[0] + input.prob[1]) ? 1 : 2;
            }
        }

        real_type w = input.weight[index];
        real_type n = pn[index];
        real_type b = pb[index];

        n = 1 / n;

        // Sampling of scattering angle
        real_type x;
        real_type y = w * generate_canonical(rng);

        if (y < fit_slice())
        {
            x = y * n * (1 + half * (n + 1) * y * (1 - (n + 2) * y / 3));
        }
        else
        {
            x = std::pow(1 - y, -n) - 1;
        }

        cost = 1 - 2 * x / (b * input.factor);

    } while (2 * generate_canonical(rng) > 1 + ipow<2>(cost) || cost < -1);

    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Scattered direction
    result.direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    return result;
}

CELER_FUNCTION
auto RayleighInteractor::evaluate_weight_and_prob(real_type      energy,
                                                  const ItemIdT& item_id) const
    -> SampleInput
{
    SampleInput input;

    Real3 a = shared_.params.data[item_id].a;
    Real3 b = shared_.params.data[item_id].b;
    Real3 n = shared_.params.data[item_id].n;

    input.factor = ipow<2>(energy * RayleighInteractor::hc_factor());

    Real3 x = b;
    axpy(input.factor, b, &x);

    Real3 prob;
    for (auto i : range(3))
    {
        input.weight[i] = (x[i] > fit_slice())
                              ? 1 - std::pow(1 + x[i], -n[i])
                              : n[i] * x[i]
                                    * (1
                                       - real_type(0.5) * (n[i] - 1) * (x[i])
                                             * (1 - (n[i] - 2) * (x[i]) / 3));

        prob[i] = input.weight[i] * a[i] / (b[i] * n[i]);
    }

    real_type inv_sum = 1 / (prob[0] + prob[1] + prob[2]);
    axpy(inv_sum, prob, &input.prob);

    return input;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

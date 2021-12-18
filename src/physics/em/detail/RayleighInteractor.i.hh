//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/IsotropicDistribution.hh"
#include "random/Selector.hh"

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
    , inc_energy_(particle.energy())
    , inc_direction_(direction)
    , element_id_(el_id)
{
    CELER_EXPECT(particle.particle_id() == shared_.gamma_id);
    CELER_EXPECT(element_id_ < shared_.params.size());
}

//---------------------------------------------------------------------------//
/*!
 * Sample the Rayleigh scattering angle using the G4LivermoreRayleighModel
 * and G4RayleighAngularGenerator of Geant4 6.10.
 */
template<class Engine>
CELER_FUNCTION Interaction RayleighInteractor::operator()(Engine& rng)
{
    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy = inc_energy_;

    SampleInput input = this->evaluate_weight_and_prob();

    const Real3& pb = shared_.params[element_id_].b;
    const Real3& pn = shared_.params[element_id_].n;

    constexpr real_type half = 0.5;
    real_type           cost;

    do
    {
        // Sample index from input.prob
        const unsigned int index = celeritas::make_selector(
            [&input](unsigned int i) { return input.prob[i]; },
            input.prob.size())(rng);

        const real_type w    = input.weight[index];
        const real_type ninv = 1 / pn[index];
        const real_type b    = pb[index];

        // Sampling of scattering angle
        real_type x;
        real_type y = w * generate_canonical(rng);

        if (y < fit_slice())
        {
            x = y * ninv
                * (1 + half * (ninv + 1) * y * (1 - (ninv + 2) * y / 3));
        }
        else
        {
            x = std::pow(1 - y, -ninv) - 1;
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
auto RayleighInteractor::evaluate_weight_and_prob() const -> SampleInput
{
    const Real3& a = shared_.params[element_id_].a;
    const Real3& b = shared_.params[element_id_].b;
    const Real3& n = shared_.params[element_id_].n;

    SampleInput input;
    input.factor = ipow<2>(units::centimeter * native_value_from(inc_energy_)
                           / (constants::c_light * constants::h_planck));

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

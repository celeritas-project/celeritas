//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/RayleighInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/RayleighData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/Selector.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply the Livermore model of Rayleigh scattering to photons.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4LivermoreRayleighModel class, as documented in section 6.2.2 of the
 * Geant4 Physics Reference (release 10.6).
 */
class RayleighInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION RayleighInteractor(RayleighRef const& shared,
                                             ParticleTrackView const& particle,
                                             Real3 const& inc_direction,
                                             ElementId element_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    RayleighRef const& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    Real3 const& inc_direction_;
    // Id of element
    ElementId element_id_;

    //// CONSTANTS ////

    //! cm/hc in the MeV energy unit
    static CELER_CONSTEXPR_FUNCTION real_type hc_factor()
    {
        return units::centimeter * native_value_from(units::MevEnergy{1.0})
               / (constants::c_light * constants::h_planck);
    }

    //! A point where the functional form of the form factor fit changes
    static CELER_CONSTEXPR_FUNCTION real_type fit_slice() { return 0.02; }

    //// HELPER TYPES ////

    //! Intermediate data for sampling input
    struct SampleInput
    {
        real_type factor{0};
        Real3 weight{0, 0, 0};
        Real3 prob{0, 0, 0};
    };

    //// HELPER FUNCTIONS ////

    //! Evaluate weights and probabilities for the angular sampling algorithm
    inline CELER_FUNCTION auto evaluate_weight_and_prob() const -> SampleInput;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
RayleighInteractor::RayleighInteractor(RayleighRef const& shared,
                                       ParticleTrackView const& particle,
                                       Real3 const& direction,
                                       ElementId el_id)

    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_direction_(direction)
    , element_id_(el_id)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.gamma);
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
    result.energy = inc_energy_;

    SampleInput input = this->evaluate_weight_and_prob();

    Real3 const& pb = shared_.params[element_id_].b;
    Real3 const& pn = shared_.params[element_id_].n;

    constexpr real_type half = 0.5;
    real_type cost;

    do
    {
        // Sample index from input.prob
        unsigned int const index = celeritas::make_selector(
            [&input](unsigned int i) { return input.prob[i]; },
            input.prob.size())(rng);

        const real_type w = input.weight[index];
        const real_type ninv = 1 / pn[index];
        const real_type b = pb[index];

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
            x = fastpow(1 - y, -ninv) - 1;
        }

        cost = 1 - 2 * x / (b * input.factor);

    } while (2 * generate_canonical(rng) > 1 + ipow<2>(cost) || cost < -1);

    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Scattered direction
    result.direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    CELER_ENSURE(result.action == Interaction::Action::scattered);
    return result;
}

CELER_FUNCTION
auto RayleighInteractor::evaluate_weight_and_prob() const -> SampleInput
{
    Real3 const& a = shared_.params[element_id_].a;
    Real3 const& b = shared_.params[element_id_].b;
    Real3 const& n = shared_.params[element_id_].n;

    SampleInput input;
    input.factor = ipow<2>(units::centimeter
                           / (constants::c_light * constants::h_planck)
                           * native_value_from(inc_energy_));

    Real3 x = b;
    axpy(input.factor, b, &x);

    Real3 prob;
    for (int i = 0; i < 3; ++i)
    {
        input.weight[i] = (x[i] > this->fit_slice())
                              ? 1 - fastpow(1 + x[i], -n[i])
                              : n[i] * x[i]
                                    * (1
                                       - (n[i] - 1) / 2 * x[i]
                                             * (1 - (n[i] - 2) / 3 * x[i]));

        prob[i] = input.weight[i] * a[i] / (b[i] * n[i]);
    }

    real_type inv_sum = 1 / (prob[0] + prob[1] + prob[2]);
    axpy(inv_sum, prob, &input.prob);

    return input;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

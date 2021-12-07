//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RBEnergySampler.i.hh
//---------------------------------------------------------------------------//
#include <cmath>

#include "base/Algorithms.hh"
#include "random/distributions/BernoulliDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 */
CELER_FUNCTION
RBEnergySampler::RBEnergySampler(const RelativisticBremNativeRef& shared,
                                 const ParticleTrackView&         particle,
                                 const CutoffView&                cutoffs,
                                 const MaterialView&              material,
                                 const ElementComponentId&        elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , gamma_cutoff_(cutoffs.energy(shared.ids.gamma))
    , dxsec_(shared, particle, material, elcomp_id)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy based on G4eBremsstrahlungRelModel
 * of the Geant4 10.7 release.
 */
template<class Engine>
CELER_FUNCTION auto RBEnergySampler::operator()(Engine& rng) -> Energy
{
    // Min and max kinetic energy limits for sampling the secondary photon
    Energy tmin = min(gamma_cutoff_, inc_energy_);
    Energy tmax = min(shared_.high_energy_limit(), inc_energy_);

    real_type density_corr = dxsec_.density_correction();
    real_type xmin         = std::log(ipow<2>(tmin.value()) + density_corr);
    real_type xrange = std::log(ipow<2>(tmax.value()) + density_corr) - xmin;

    real_type gamma_energy{0};
    real_type dsigma{0};

    do
    {
        gamma_energy = std::sqrt(max(
            real_type(0),
            std::exp(xmin + generate_canonical(rng) * xrange) - density_corr));
        dsigma       = dxsec_(gamma_energy);
    } while (dsigma < dxsec_.maximum_value() * generate_canonical(rng));

    return units::MevEnergy{gamma_energy};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

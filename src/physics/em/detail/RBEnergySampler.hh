//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RBEnergySampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Algorithms.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/ReciprocalDistribution.hh"
#include "sim/Types.hh"

#include "PhysicsConstants.hh"
#include "RBDiffXsCalculator.hh"
#include "RelativisticBremData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy from the relativistic model.
 */
class RBEnergySampler
{
  public:
    //!@{
    //! Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION RBEnergySampler(const RelativisticBremRef& shared,
                                          const ParticleTrackView&   particle,
                                          const CutoffView&          cutoffs,
                                          const MaterialView&        material,
                                          const ElementComponentId& elcomp_id);

    // Sample the bremsstrahlung photon energy with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// DATA ////

    // Differential cross section calcuator
    RBDiffXsCalculator calc_dxsec_;
    // Square of minimum of incident particle energy and cutoff
    real_type tmin_sq_;
    // Square of production cutoff for gammas
    real_type tmax_sq_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 */
CELER_FUNCTION
RBEnergySampler::RBEnergySampler(const RelativisticBremRef& shared,
                                 const ParticleTrackView&   particle,
                                 const CutoffView&          cutoffs,
                                 const MaterialView&        material,
                                 const ElementComponentId&  elcomp_id)
    : calc_dxsec_(shared, particle.energy(), material, elcomp_id)
{
    // Min and max kinetic energy limits for sampling the secondary photon
    real_type gamma_cutoff = value_as<Energy>(cutoffs.energy(shared.ids.gamma));
    real_type inc_energy   = value_as<Energy>(particle.energy());

    tmin_sq_ = ipow<2>(min(gamma_cutoff, inc_energy));
    tmax_sq_ = ipow<2>(min(value_as<Energy>(high_energy_limit()), inc_energy));

    CELER_ENSURE(tmax_sq_ >= tmin_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy based on G4eBremsstrahlungRelModel
 * of the Geant4 10.7 release.
 */
template<class Engine>
CELER_FUNCTION auto RBEnergySampler::operator()(Engine& rng) -> Energy
{
    real_type density_corr = calc_dxsec_.density_correction();
    ReciprocalDistribution<real_type> sample_exit_esq(tmin_sq_ + density_corr,
                                                      tmax_sq_ + density_corr);

    // Sampled energy and corresponding cross section for rejection
    real_type gamma_energy{0};
    real_type dsigma{0};

    do
    {
        gamma_energy = std::sqrt(sample_exit_esq(rng) - density_corr);
        dsigma       = calc_dxsec_(Energy{gamma_energy});
    } while (!BernoulliDistribution(dsigma / calc_dxsec_.maximum_value())(rng));

    return Energy{gamma_energy};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

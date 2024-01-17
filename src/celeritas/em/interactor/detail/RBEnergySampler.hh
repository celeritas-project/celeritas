//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/RBEnergySampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/xs/RBDiffXsCalculator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/ReciprocalDistribution.hh"

#include "PhysicsConstants.hh"

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
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION RBEnergySampler(RelativisticBremRef const& shared,
                                          Energy const& inc_energy,
                                          CutoffView const& cutoffs,
                                          MaterialView const& material,
                                          ElementComponentId const& elcomp_id);

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
RBEnergySampler::RBEnergySampler(RelativisticBremRef const& shared,
                                 Energy const& inc_energy,
                                 CutoffView const& cutoffs,
                                 MaterialView const& material,
                                 ElementComponentId const& elcomp_id)
    : calc_dxsec_(shared, inc_energy, material, elcomp_id)
{
    // Min and max kinetic energy limits for sampling the secondary photon
    real_type gamma_cutoff = value_as<Energy>(cutoffs.energy(shared.ids.gamma));
    tmin_sq_ = ipow<2>(min(gamma_cutoff, inc_energy.value()));
    tmax_sq_ = ipow<2>(min(value_as<Energy>(detail::high_energy_limit()),
                           inc_energy.value()));

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
        dsigma = calc_dxsec_(Energy{gamma_energy});
    } while (!BernoulliDistribution(dsigma / calc_dxsec_.maximum_value())(rng));

    return Energy{gamma_energy};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

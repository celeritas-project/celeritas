//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/RBEnergySampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/ReciprocalDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/xs/RBDiffXsCalculator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "PhysicsConstants.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy from the relativistic model.
 *
 * Based on \c G4eBremsstrahlungRelModel of the Geant4 10.7 release.
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
    inline CELER_FUNCTION
    RBEnergySampler(NativeCRef<RelativisticBremData> const& shared,
                    ParticleTrackView const& particle,
                    CutoffView const& cutoffs,
                    MaterialView const& material,
                    ElementComponentId elcomp_id);

    // Sample the bremsstrahlung photon energy with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// DATA ////

    // Differential cross section calculator
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
RBEnergySampler::RBEnergySampler(NativeCRef<RelativisticBremData> const& shared,
                                 ParticleTrackView const& particle,
                                 CutoffView const& cutoffs,
                                 MaterialView const& material,
                                 ElementComponentId elcomp_id)
    : calc_dxsec_(shared, particle, material, elcomp_id)
    , tmin_sq_(ipow<2>(value_as<Energy>(
          min(cutoffs.energy(shared.ids.gamma), particle.energy()))))
    , tmax_sq_(ipow<2>(value_as<Energy>(
          min(detail::high_energy_limit(), particle.energy()))))
{
    CELER_ENSURE(tmax_sq_ >= tmin_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy.
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
    } while (RejectionSampler(dsigma, calc_dxsec_.maximum_value())(rng));

    return Energy{gamma_energy};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

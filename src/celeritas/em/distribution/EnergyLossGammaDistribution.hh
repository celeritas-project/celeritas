//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossGammaDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/GammaDistribution.hh"

#include "EnergyLossHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample energy loss from a gamma distribution.
 *
 * This model is valid for heavy particles with small mean losses (less than 2
 * Bohr standard deviations). It is a special case of
 * \c EnergyLossGaussianDistribution.
 */
class EnergyLossGammaDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using EnergySq = RealQuantity<UnitProduct<units::Mev, units::Mev>>;
    //!@}

  public:
    // Construct from distribution parameters
    inline CELER_FUNCTION
    EnergyLossGammaDistribution(Energy mean_loss, EnergySq bohr_var);

    // Construct from helper-calculated data
    explicit inline CELER_FUNCTION
    EnergyLossGammaDistribution(EnergyLossHelper const& helper);

    //! Sample energy loss according to the distribution
    template<class Generator>
    CELER_FUNCTION Energy operator()(Generator& rng)
    {
        return Energy{sample_gamma_(rng)};
    }

  private:
    using GammaDist = GammaDistribution<real_type>;

    GammaDist sample_gamma_;

    static inline CELER_FUNCTION GammaDist build_gamma(real_type mean,
                                                       real_type stddev);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from distribution parameters.
 *
 * The model is only valid (rather, used in Geant4) for mean < 2 * stddev, but
 * it is allowable to construct the sampler explicitly but outside this range,
 * for analysis purposes.
 */
CELER_FUNCTION
EnergyLossGammaDistribution::EnergyLossGammaDistribution(Energy mean_loss,
                                                         EnergySq bohr_var)
    : sample_gamma_(EnergyLossGammaDistribution::build_gamma(mean_loss.value(),
                                                             bohr_var.value()))
{
    CELER_EXPECT(mean_loss > zero_quantity());
    CELER_EXPECT(bohr_var > zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from helper-calculated data.
 */
CELER_FUNCTION EnergyLossGammaDistribution::EnergyLossGammaDistribution(
    EnergyLossHelper const& helper)
    : EnergyLossGammaDistribution(helper.mean_loss(), helper.bohr_variance())
{
    CELER_ASSERT(helper.model() == EnergyLossFluctuationModel::gamma);
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to construct gamma distribution.
 */
CELER_FUNCTION auto
EnergyLossGammaDistribution::build_gamma(real_type mean, real_type var)
    -> GammaDist
{
    real_type k = ipow<2>(mean) / var;
    return GammaDist{k, mean / k};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

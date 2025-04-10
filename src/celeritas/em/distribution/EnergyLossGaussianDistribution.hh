//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossGaussianDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/NormalDistribution.hh"

#include "EnergyLossHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample energy loss from a Gaussian distribution.
 *
 * In a thick absorber, the total energy transfer is a result of many small
 * energy losses from a large number of collisions. The central limit theorem
 * applies, and the energy loss fluctuations can be described by a Gaussian
 * distribution. See section 7.3.1 of the Geant4 Physics Reference Manual and
 * GEANT3 PHYS332 section 2.3.
 *
 * The Gaussian approximation is valid for heavy particles and in the
 * regime \f$ \kappa = \xi / T_\textrm{max} > 10 \f$.
 * Fluctuations of the unrestricted energy loss
 * follow a Gaussian distribution if \f$ \Delta E > \kappa T_{max} \f$,
 * where \f$ T_{max} \f$ is the maximum energy transfer (PHYS332 section
 * 2). For fluctuations of the \em restricted energy loss, the condition is
 * modified to \f$ \Delta E > \kappa T_{c} \f$ and \f$ T_{max} \le 2 T_c
 * \f$, where \f$ T_c \f$ is the delta ray cutoff energy (PRM Eq. 7.6-7.7).
 */
class EnergyLossGaussianDistribution
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
    EnergyLossGaussianDistribution(Energy mean_loss, Energy bohr_stddev);

    // Construct from helper distribution parameters
    inline CELER_FUNCTION
    EnergyLossGaussianDistribution(Energy mean_loss, EnergySq bohr_var);

    // Construct from helper-calculated data
    explicit inline CELER_FUNCTION
    EnergyLossGaussianDistribution(EnergyLossHelper const& helper);

    // Sample energy loss according to the distribution
    template<class Generator>
    inline CELER_FUNCTION Energy operator()(Generator& rng);

  private:
    real_type const max_loss_;
    NormalDistribution<real_type> sample_normal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from mean/stddev.
 *
 * This formulation is used internally by the Urban distribution.
 */
CELER_FUNCTION EnergyLossGaussianDistribution::EnergyLossGaussianDistribution(
    Energy mean_loss, Energy bohr_stddev)
    : max_loss_(2 * mean_loss.value())
    , sample_normal_(mean_loss.value(), bohr_stddev.value())

{
    CELER_EXPECT(mean_loss > zero_quantity());
    CELER_EXPECT(bohr_stddev > zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from distribution parameters.
 *
 * The mean loss is the energy lost over the step, and the standard deviation
 * is the square root of Bohr's variance (PRM Eq. 7.8). For thick absorbers,
 * the straggling function approaches a Gaussian distribution with this
 * standard deviation.
 */
CELER_FUNCTION EnergyLossGaussianDistribution::EnergyLossGaussianDistribution(
    Energy mean_loss, EnergySq bohr_var)
    : EnergyLossGaussianDistribution{mean_loss,
                                     Energy{std::sqrt(bohr_var.value())}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from helper-calculated data.
 */
CELER_FUNCTION EnergyLossGaussianDistribution::EnergyLossGaussianDistribution(
    EnergyLossHelper const& helper)
    : EnergyLossGaussianDistribution{helper.mean_loss(), helper.bohr_variance()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample energy loss according to the distribution.
 */
template<class Generator>
CELER_FUNCTION auto
EnergyLossGaussianDistribution::operator()(Generator& rng) -> Energy
{
    real_type result;
    do
    {
        result = sample_normal_(rng);
    } while (result <= 0 || result > max_loss_);
    return Energy{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

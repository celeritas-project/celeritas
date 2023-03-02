//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/MscStepFromGeo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert the "geometrical" path traveled to the true path with MSC.
 *
 * The "true" step length is the physical path length taken along the
 * geometrical path length (which is straight without magnetic fields or curved
 * with them), accounting for the extra distance taken between
 * along-step elastic collisions.
 *
 * The transformation can be written as
 * \f[
 *     t(z) = \langle t \rangle = -\lambda_{1} \log(1 - \frac{z}{\lambda_{1}})
 * \f]
 * or \f$ t(z) = \frac{1}{\alpha} [ 1 - (1-\alpha w z)^{1/w}] \f$ if the
 * geom path is small, where \f$ w = 1 + \frac{1}{\alpha \lambda_{10}}\f$.
 *
 * \param true_path the proposed step before transportation.
 * \param gstep the proposed step after transportation.
 * \param alpha variable from UrbanMscStepLimit.
 */
class MscStepFromGeo
{
  public:
    //!@{
    //! \name Type aliases
    using MscParameters = UrbanMscParameters;
    //!@}

  public:
    // Construct with path length data
    inline CELER_FUNCTION MscStepFromGeo(MscParameters const& params_,
                                         MscStep const& step,
                                         real_type range,
                                         real_type lambda);

    // Calculate the true path
    inline CELER_FUNCTION real_type operator()(real_type gstep) const;

  private:
    // Urban MSC parameters
    MscParameters const& params_;
    //! Physical path length limited and transformed by MSC
    real_type true_step_;
    //! Scaled slope of the MFP
    real_type alpha_;
    //! Range for checking edge cases
    real_type range_;
    //! MFP at the start of step
    real_type lambda_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from MSC data.
 */
CELER_FUNCTION MscStepFromGeo::MscStepFromGeo(MscParameters const& params,
                                              MscStep const& step,
                                              real_type range,
                                              real_type lambda)
    : params_(params)
    , true_step_(step.true_path)
    , alpha_(step.alpha)
    , range_(range)
    , lambda_(lambda)
{
    CELER_EXPECT(lambda_ > 0);
    CELER_EXPECT(true_step_ <= range_);
    // TODO: expect that the "skip sampling" conditions are false.
}

//---------------------------------------------------------------------------//
/*!
 * Calculate and return the true path from the input "geometrical" step.
 *
 * This should only be applied if the step was geometry-limited. Otherwise, the
 * original interaction length is correct.
 */
CELER_FUNCTION real_type MscStepFromGeo::operator()(real_type gstep) const
{
    CELER_EXPECT(gstep >= 0 && gstep <= true_step_);

    if (gstep < params_.min_step())
    {
        // geometrical path length = true path length for a very small step
        return gstep;
    }

    // NOTE: add && !insideskin if the UseDistanceToBoundary algorithm is used
    if (gstep <= lambda_ * params_.tau_small)
    {
        // Very small distance to collision (less than tau_small paths)
        return gstep;
    }

    real_type tstep = [&] {
        if (alpha_ == MscStep::small_step_alpha())
        {
            // Cross section was assumed to be constant over the step:
            // z = lambda * (1 - exp(-tau))
            real_type result = -lambda_ * std::log(1 - gstep / lambda_);
            CELER_ENSURE(result >= gstep && result <= true_step_);
            return result;
        }

        real_type w = 1 + 1 / (alpha_ * lambda_);
        real_type x = alpha_ * w * gstep;  // = (1 - (1 - alpha * true)^w)
        if (CELER_UNLIKELY(x > 1))
        {
            // Range-limited step results in x = 1, which gives the correct
            // result in the inverted equation below for alpha = 1/range.
            // x >= 1 corresponds to the stopping MFP being <= 0: zero is
            // correct when the step is the range, but the MFP will never be
            // zero except for numerical error.
            CELER_ASSERT(x < 1 + 100 * numeric_limits<real_type>::epsilon());
            CELER_ENSURE(range_ >= gstep && range_ <= true_step_);
            return range_;
        }
        // Invert Eq. 8.10 exactly
        real_type result = (1 - fastpow(1 - x, 1 / w)) / alpha_;
        CELER_ENSURE(result >= gstep && result <= true_step_);
        return result;
    }();

    // No less than the geometry path if effectively no scattering
    // took place; no more than the original calculated true path if the step
    // is path-limited.
    CELER_ENSURE(tstep >= gstep);
    CELER_ENSURE(tstep <= true_step_);
    return tstep;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

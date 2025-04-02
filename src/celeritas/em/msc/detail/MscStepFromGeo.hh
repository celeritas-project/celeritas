//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/MscStepFromGeo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
namespace detail
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
        // Geometrical path length is true path length for a very small step
        return gstep;
    }

    real_type tstep = [&] {
        if (alpha_ == MscStep::small_step_alpha())
        {
            // Cross section was assumed to be constant over the step:
            // z = lambda * (1 - exp(-g / lambda))
            // => g = -lambda * log(1 - g / lambda)
            real_type tstep = -lambda_ * std::log1p(-gstep / lambda_);
            if (tstep < params_.min_step())
            {
                // Geometrical path length = true path length for a very small
                // step
                return gstep;
            }
            return tstep;
        }

        real_type w = 1 + 1 / (alpha_ * lambda_);
        real_type x = alpha_ * w * gstep;  // = (1 - (1 - alpha * true)^w)
        // Range-limited step results in x = 1, which gives the correct
        // result in the inverted equation below for alpha = 1/range.
        // x >= 1 corresponds to the stopping MFP being <= 0: x=1 meaning
        // ending MFP of zero is correct when the step is the range. Precision
        // loss means this conversion also may result in x > 1, which we guard
        // against.
        x = min(x, real_type(1));
        // Near x=1, (1 - (1-x)^(1/w)) suffers from numerical precision loss;
        // the maximum value of the expression should be range_.
        // TODO: we should use the action ID to avoid applying this
        // transformation if not range-limited.
        real_type temp = 1 - fastpow(1 - x, 1 / w);
        real_type result = temp / alpha_;
        return min(result, range_);
    }();

    // The result can be no less than the geometry path (effectively no
    // scattering took place) and no more than the original calculated true
    // path (if the step is path-limited).
    // The result can be slightly outside the bounds due to numerical
    // imprecision and edge cases:
    // - a few ULP due to inexactness of log1p/expm1
    // - small travel distance results in roundoff error inside (1 -
    //   alpha/true)^w
    // - gstep is just above min_step so that the updated recalculation of
    //   tstep is just below min_step
    return clamp(tstep, gstep, true_step_);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

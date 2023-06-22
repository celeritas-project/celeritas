//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WokviXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/interactor/detail/WokviStateHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the ratio of Mott cross section to the Rutherford cross section.
 */
class MottXsCalculator
{
  public:
    // Construct with state data
    inline CELER_FUNCTION
    MottXsCalculator(detail::WokviStateHelper const& state);

    // Ratio of Mott and Rutherford cross sections
    inline CELER_FUNCTION real_type operator()(real_type fcos_t) const;

  private:
    detail::WokviStateHelper const& state_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state data
 */
CELER_FUNCTION
MottXsCalculator::MottXsCalculator(detail::WokviStateHelper const& state)
    : state_(state)
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute the ratio of Mott to Rutherford cross sections.
 * The parameter fcos_t is equivalent to
 *      sqrt(1 - cos(theta))
 * where theta is the scattered angle in the z-aligned momentum frame.
 */
CELER_FUNCTION
real_type MottXsCalculator::operator()(real_type fcos_t) const
{
    real_type ratio = 0;
    const real_type shift = 0.7181228;
    const real_type beta0 = sqrt(1.0 / state_.inv_beta_sq) - shift;

    // Construct [0,5] powers of beta0
    real_type b0 = 1.0;
    real_type b[6];
    for (int i = 0; i < 6; i++) {
        b[i] = b0;
        b0 *= beta0;
    }

    // Compute the ratio
    real_type f0 = 1.0;
    for (int j = 0; j <= 4; j++) {
        real_type a = 0.0;
        for (int k = 0; k < 6; k++) {
            a += state_.element_data.mott_coeff[j][k] * b[k];
        }
        ratio += a * f0;
        f0 *= fcos_t;
    }

    return ratio;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

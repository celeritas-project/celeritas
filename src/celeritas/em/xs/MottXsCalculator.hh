//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/MottXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/em/data/WentzelData.hh"

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
    MottXsCalculator(WentzelElementData const& element_data,
                     real_type inc_energy,
                     real_type inc_mass);

    // Ratio of Mott and Rutherford cross sections
    inline CELER_FUNCTION real_type operator()(real_type cos_t) const;

  private:
    WentzelElementData const& element_data_;
    real_type beta_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state data
 */
CELER_FUNCTION
MottXsCalculator::MottXsCalculator(WentzelElementData const& element_data,
                                   real_type inc_energy,
                                   real_type inc_mass)
    : element_data_(element_data)
{
    real_type const inc_mom = sqrt(inc_energy * (inc_energy + 2 * inc_mass));
    beta_ = inc_mom / (inc_energy + inc_mass);
    CELER_EXPECT(beta_ < 1);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the ratio of Mott to Rutherford cross sections.
 *
 * The parameter cos_theta is the cosine of the
 * scattered angle in the z-aligned momentum frame.
 *
 * For 1 <= Z <= 92, an interpolated expression is used [PRM 8.48].
 */
CELER_FUNCTION
real_type MottXsCalculator::operator()(real_type cos_theta) const
{
    const real_type fcos_t = sqrt(1 - cos_theta);

    // Mean velocity of electrons between ~KeV and 900 MeV
    const real_type beta_shift = 0.7181228;
    const real_type beta0 = beta_ - beta_shift;

    // Construct [0,5] powers of beta0
    real_type b[6];
    b[0] = 1.0;
    for (int i = 1; i < 6; i++)
    {
        b[i] = beta0 * b[i - 1];
    }

    // Compute the ratio, summing over powers of fcos_t
    real_type f0 = 1.0;
    real_type ratio = 0;
    for (int j = 0; j <= 4; j++)
    {
        // Calculate the a_j coefficient
        real_type a = 0.0;
        for (int k = 0; k <= 5; k++)
        {
            a += element_data_.mott_coeff[j][k] * b[k];
        }
        // Sum in power series of fcos_t
        ratio += a * f0;
        f0 *= fcos_t;
    }

    return ratio;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

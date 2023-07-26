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
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/em/data/WentzelData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the ratio of Mott cross section to the Rutherford cross section.
 *
 * The ratio is an interpolated approximation developed in
 * T. Lijian, H. Quing and L. Zhengming, Radiat. Phys. Chem. 45 (1995),
 *   235-245
 * and described in the Geant Physics Reference Manual [PRM] (Release 1.11)
 * section 8.4.
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

    template<class T>
    inline CELER_FUNCTION void generate_powers(T& powers, real_type x) const;
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
    CELER_EXPECT(cos_theta >= -1 && cos_theta <= 1);

    // (Exponent) Base for theta powers
    const real_type fcos_t = sqrt(1 - cos_theta);

    // Mean velocity of electrons between ~KeV and 900 MeV
    const real_type beta_shift = 0.7181228;

    // (Exponent) Base for beta powers
    const real_type beta0 = beta_ - beta_shift;

    // Construct arrays of powers
    WentzelElementData::BetaArray beta_powers;
    generate_powers(beta_powers, beta0);

    WentzelElementData::ThetaArray theta_powers;
    generate_powers(theta_powers, fcos_t);

    // Inner product the arrays of powers with the coefficient matrix
    WentzelElementData::ThetaArray theta_coeffs;
    for (auto i : range(theta_coeffs.size()))
    {
        theta_coeffs[i] = dot_product(element_data_.mott_coeff[i], beta_powers);
    }
    return dot_product(theta_coeffs, theta_powers);
}

//---------------------------------------------------------------------------//
/*!
 * Fills the array with powers of x from \f$ x^0, x^1, ..., x^{N-1} \f$,
 * where \f$ N \f$ is the size of the array.
 */
template<class T>
CELER_FUNCTION void
MottXsCalculator::generate_powers(T& powers, real_type x) const
{
    powers[0] = 1;
    for (size_type i : range(1u, powers.size()))
    {
        powers[i] = x * powers[i - 1];
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

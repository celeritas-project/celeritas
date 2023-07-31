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
#include "celeritas/grid/PolyEvaluator.hh"

namespace celeritas
{

// TEST

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
    real_type inc_mom = sqrt(inc_energy * (inc_energy + 2 * inc_mass));
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
    real_type fcos_t = sqrt(1 - cos_theta);

    // Mean velocity of electrons between ~KeV and 900 MeV
    const real_type beta_shift = 0.7181228;

    // (Exponent) Base for beta powers
    real_type beta0 = beta_ - beta_shift;

    // Evaluate polynomial of powers of beta0 and fcos_t
    WentzelElementData::ThetaArray theta_coeffs;
    for (auto i : range(theta_coeffs.size()))
    {
        theta_coeffs[i] = PolyEvaluator{element_data_.mott_coeff[i]}(beta0);
    }
    return PolyEvaluator{theta_coeffs}(fcos_t);
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
    for (size_type i : range(size_type{1}, powers.size()))
    {
        powers[i] = x * powers[i - 1];
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

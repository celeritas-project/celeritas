//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/MottRatioCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Calculates the ratio of Mott cross section to the Rutherford cross section.
 *
 * This ratio is an adjustment of the cross section from a purely classical
 * treatment of a point nucleus in an electronic cloud (Rutherford scattering)
 * to a quantum mechanical treatment. The implementation is an interpolated
 * approximation developed in \citet{lijian-mott-1995,
 * https://doi.org/10.1016/0969-806X(94)00063-8} and described in \citet{g4prm,
 * https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/index.html}
 * section 8.4.
 *
 * The input argument \c cos_theta is the cosine of the scattered angle in the
 * z-aligned momentum frame.
 */
class MottRatioCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using MottCoeffMatrix = MottElementData::MottCoeffMatrix;
    //!@}

  public:
    // Construct with state data
    inline CELER_FUNCTION
    MottRatioCalculator(MottCoeffMatrix const& coeffs, real_type beta);

    // Ratio of Mott and Rutherford cross sections
    inline CELER_FUNCTION real_type operator()(real_type cos_t) const;

  private:
    MottCoeffMatrix const& coeffs_;
    real_type beta_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state data.
 */
CELER_FUNCTION
MottRatioCalculator::MottRatioCalculator(MottCoeffMatrix const& coeffs,
                                         real_type beta)
    : coeffs_(coeffs), beta_(beta)
{
    CELER_EXPECT(0 <= beta_ && beta_ < 1);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the ratio of Mott to Rutherford cross sections.
 */
CELER_FUNCTION
real_type MottRatioCalculator::operator()(real_type cos_theta) const
{
    CELER_EXPECT(cos_theta >= -1 && cos_theta <= 1);

    // (Exponent) Base for theta powers
    real_type fcos_t = std::sqrt(1 - cos_theta);

    // Mean velocity of electrons between ~KeV and 900 MeV
    real_type const beta_shift = 0.7181228;

    // (Exponent) Base for beta powers
    real_type beta0 = beta_ - beta_shift;

    // Evaluate polynomial of powers of beta0 and fcos_t
    MottElementData::ThetaArray theta_coeffs;
    for (auto i : range(theta_coeffs.size()))
    {
        theta_coeffs[i] = PolyEvaluator(coeffs_[i])(beta0);
    }
    real_type result = PolyEvaluator(theta_coeffs)(fcos_t);
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

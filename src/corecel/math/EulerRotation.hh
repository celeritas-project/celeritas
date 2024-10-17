//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/EulerRotation.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform a vector rotation in eucledian space using Euler angles.
 *
 * Constructor takes the three Euler angles \em phi , \em theta , and \em psi ,
 * which are used for 3 axis rotations, denoted as B, C, and D. The final
 * general rotation, \f$ A = B \times C \times D \f$, is applied via the
 * \c operator() .
 *
 * The Euler angles follow the x-convention, where:
 * - \em phi is the angle of the first rotation (D), on the \em z-axis
 * - \em theta is the angle of the second rotation (C), on the \em x'-axis
 * (which is the rotated x-axis)
 * - \em psi is the angle of the third rotation (B), on the \em z'-axis (which
 * is the rotate z-axis)
 *
 * \note For details see https://mathworld.wolfram.com/EulerAngles.html
 */
class EulerRotation
{
  public:
    // Construct with Euler angles in radians
    inline CELER_FUNCTION
    EulerRotation(real_type phi, real_type theta, real_type psi);

    // Return a rotated result given an input vector
    inline CELER_FUNCTION Real3 operator()(Real3 const& vector);

  private:
    // General rotation matrix (A) elements
    real_type a11_, a12_, a13_;
    real_type a21_, a22_, a23_;
    real_type a31_, a32_, a33_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct general rotation matrix (A) with provided Euler angles [radians].
 */
CELER_FUNCTION
EulerRotation::EulerRotation(real_type phi, real_type theta, real_type psi)
{
    using std::cos;
    using std::sin;

    CELER_EXPECT(phi >= 0 && phi <= 2 * constants::pi);
    CELER_EXPECT(theta >= 0 && theta <= constants::pi);
    CELER_EXPECT(psi >= 0 && psi <= 2 * constants::pi);

    real_type const sin_phi = sin(phi);
    real_type const cos_phi = cos(phi);
    real_type const sin_theta = sin(theta);
    real_type const cos_theta = cos(theta);
    real_type const sin_psi = sin(psi);
    real_type const cos_psi = cos(psi);

    a11_ = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi;
    a12_ = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi;
    a13_ = sin_psi * sin_theta;

    a21_ = -sin_psi * cos_phi - cos_theta * sin_phi * cos_psi;
    a22_ = -sin_psi * sin_phi + cos_theta * cos_phi * cos_psi;
    a23_ = cos_psi * sin_theta;

    a31_ = sin_theta * sin_phi;
    a32_ = -sin_theta * cos_phi;
    a33_ = cos_theta;
}

//---------------------------------------------------------------------------//
/*!
 * Return the rotated result from the provided input vector.
 */
CELER_FUNCTION Real3 EulerRotation::operator()(Real3 const& vector)
{
    return {a11_ * vector[0] + a12_ * vector[1] + a13_ * vector[2],
            a21_ * vector[0] + a22_ * vector[1] + a23_ * vector[2],
            a31_ * vector[0] + a32_ * vector[1] + a33_ * vector[2]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

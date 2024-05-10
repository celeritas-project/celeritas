//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/Transformation.cc
//---------------------------------------------------------------------------//
#include "Transformation.hh"

#include <cmath>

#include "corecel/math/SoftEqual.hh"
#include "orange/MatrixUtils.hh"

#include "Translation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct by inverting another transformation.
 */
Transformation Transformation::from_inverse(Mat3 const& rot, Real3 const& trans)
{
    // Transpose the rotation
    Mat3 const rinv = make_transpose(rot);

    // Calculate the updated position
    Real3 tinv = gemv(real_type{-1}, rinv, trans, real_type{0}, {});
    return Transformation{rinv, tinv};
}

//---------------------------------------------------------------------------//
/*!
 * Construct as an identity transform.
 */
Transformation::Transformation() : Transformation{Translation{}} {}

//---------------------------------------------------------------------------//
/*!
 * Construct and check the input.
 *
 * The input rotation matrix should be an orthonormal matrix. Its determinant
 * is 1 if not reflecting (proper) or -1 if reflecting (improper).  It is the
 * caller's job to ensure a user-provided low-precision matrix is
 * orthonormal: see \c celeritas::orthonormalize . (Add \c CELER_VALIDATE to
 * the calling code if constructing a transformation matrix from user input or
 * a suspect source.)
 */
Transformation::Transformation(Mat3 const& rot, Real3 const& trans)
    : rot_(rot), tra_(trans)
{
    CELER_EXPECT(soft_equal(std::fabs(determinant(rot_)), real_type(1)));
}

//---------------------------------------------------------------------------//
/*!
 * Promote from a translation.
 */
Transformation::Transformation(Translation const& tr)
    : rot_{Real3{1, 0, 0}, Real3{0, 1, 0}, Real3{0, 0, 1}}
    , tra_{tr.translation()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the inverse during preprocessing.
 */
Transformation Transformation::calc_inverse() const
{
    return Transformation::from_inverse(rot_, tra_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

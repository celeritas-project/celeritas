//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/Transformation.cc
//---------------------------------------------------------------------------//
#include "Transformation.hh"

#include <cmath>

#include "corecel/math/ArraySoftUnit.hh"
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
 * Construct with rotation and translation.
 */
Transformation::Transformation(Mat3 const& rot, Real3 const& trans)
    : rot_(rot), tra_(trans)
{
    CELER_EXPECT(std::all_of(data().begin(), data().end(), [](real_type v) {
        return !std::isnan(v);
    }));
    if (CELER_UNLIKELY(this->calc_properties().scales))
    {
        CELER_NOT_IMPLEMENTED("transforms with scaling");
    }
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
/*!
 * Calculate properties about the matrix.
 */
Transformation::Properties Transformation::calc_properties() const
{
    auto det = determinant(rot_);

    Properties result;
    result.reflects = det < 0;
    result.scales = !std::all_of(rot_.begin(), rot_.end(), [](Real3 const& row) {
        return is_soft_unit_vector(row);
    });
    CELER_ENSURE(soft_equal(std::fabs(det), real_type{1}) || result.scales);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

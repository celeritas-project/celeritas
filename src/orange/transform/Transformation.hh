//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/Transformation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayOperators.hh"
#include "orange/MatrixUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/Types.hh"

namespace celeritas
{
class Translation;
class SignedPermutation;

//---------------------------------------------------------------------------//
/*!
 * Apply transformations with rotation.
 *
 * \note The nomenclature in this class assumes the translation vector and
 * rotation matrix given represent "daughter-to-parent"! This is because we
 * think of rotations being with respect to the daughter's origin rather than
 * the parent's.
 *
 * This class enables transforms between daughter and parent coordinate
 * system. The transfer from a daughter into a parent system ("up" in a
 * hierarchy of universes) is
 * \f[
   \mathbf{r}_p = \mathbf{R}\mathbf{r}_d + \mathbf{t}\:,
 * \f]
 * Where the subscripts \e p,d refer to the parent and daughter coordinate
 * systems, respectively.  The vector \b t is a translation vector.  To go
 * from the parent into the daughter system ("down" in a universe hierarchy) we
 * apply the inverse:
 * \f[
   \mathbf{r}_d = \mathbf{R}^T(\mathbf{r}_p - \mathbf{t})\:.
 * \f]
 * where the transpose of \b R is equal to its inverse because the matrix is
 * unitary.
 *
 * The rotation matrix is indexed with C ordering, [i][j].
 */
class Transformation
{
  public:
    //@{
    //! \name Type aliases
    using StorageSpan = Span<real_type const, 12>;
    using Mat3 = SquareMatrixReal3;
    //@}

    //! Transformation type identifier
    static CELER_CONSTEXPR_FUNCTION TransformType transform_type()
    {
        return TransformType::transformation;
    }

  public:
    //// CONSTRUCTORS ////

    // Construct and check the input
    Transformation(Mat3 const& rot, Real3 const& trans);

    // Construct from an identity transformation + translation
    Transformation();

    // Promote from a translation
    explicit Transformation(Translation const&);

    // Promote from a signed permutation
    explicit Transformation(SignedPermutation const&);

    // Construct inline from storage
    explicit inline CELER_FUNCTION Transformation(StorageSpan);

    //// ACCESSORS ////

    //! Rotation matrix
    CELER_FORCEINLINE_FUNCTION Mat3 const& rotation() const { return rot_; }

    //! Translation vector
    CELER_FORCEINLINE_FUNCTION Real3 const& translation() const
    {
        return tra_;
    }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&rot_[0][0], 12}; }

    //// CALCULATION ////

    // Transform from daughter to parent
    [[nodiscard]] inline CELER_FUNCTION Real3
    transform_up(Real3 const& pos) const;

    // Transform from parent to daughter
    [[nodiscard]] inline CELER_FUNCTION Real3
    transform_down(Real3 const& parent_pos) const;

    // Rotate from daughter to parent
    [[nodiscard]] inline CELER_FUNCTION Real3 rotate_up(Real3 const& dir) const;

    // Rotate from parent to daughter
    [[nodiscard]] inline CELER_FUNCTION Real3
    rotate_down(Real3 const& parent_dir) const;

  private:
    Mat3 rot_;
    Real3 tra_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct inline from storage.
 */
CELER_FUNCTION Transformation::Transformation(StorageSpan s)
    : rot_{Real3{s[0], s[1], s[2]},
           Real3{s[3], s[4], s[5]},
           Real3{s[6], s[7], s[8]}}
    , tra_{s[9], s[10], s[11]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Transform from daughter to parent.
 *
 * Apply the rotation matrix, add the translation.
 */
CELER_FUNCTION Real3 Transformation::transform_up(Real3 const& pos) const
{
    return gemv(real_type{1}, rot_, pos, real_type{1}, tra_);
}

//---------------------------------------------------------------------------//
/*!
 * Transform from parent to daughter.
 *
 * Subtract the translation, then apply the inverse of the rotation matrix (its
 * transpose).
 */
CELER_FUNCTION Real3 Transformation::transform_down(Real3 const& pos) const
{
    return gemv(matrix::transpose, rot_, pos - tra_);
}

//---------------------------------------------------------------------------//
/*!
 * Rotate from daughter to parent.
 */
CELER_FUNCTION Real3 Transformation::rotate_up(Real3 const& d) const
{
    return gemv(rot_, d);
}

//---------------------------------------------------------------------------//
/*!
 * Rotate from parent to daughter.
 */
CELER_FUNCTION Real3 Transformation::rotate_down(Real3 const& d) const
{
    return gemv(matrix::transpose, rot_, d);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

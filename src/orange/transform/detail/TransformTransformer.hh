//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/detail/TransformTransformer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"
#include "orange/MatrixUtils.hh"

#include "../NoTransformation.hh"
#include "../Transformation.hh"
#include "../Translation.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply a transformation to another transformation.
 *
 * An instance of this class applies a daughter-to-parent translation to
 * another transformation: the translation needed to relocate a "lower"
 * universe to a new coordinate system in the "higher" universe.
 *
 * Given a transform operator \b T defined \f[
   \mathbf{T}x = \mathbf{R} x + t
 * \f]
 * this operation returns a new transform
 * \f[
   \mathbf{T}' = \mathbf{T}_A \mathbf{T}_B
 * \f]
 * where \f$\mathbf{T}_A\f$ is the constructor argument
 * (the operator itself), \b T_B is the argument passed to the call operator,
 * and the \f$\mathbf{T}'\f$ is the return value.
 * The new transform has rotation
 * \f[
   \mathbf{R}' = \mathbf{R}_A\mathbf{R}_B
 * \f]
 * and translation
 * \f[
   \mathbf{t}' = \mathbf{R}_A\mathbf{t}_B + \mathbf{t}_A
 * \f]
 */
class TransformTransformer
{
  public:
    //@{
    //! \name Type aliases
    using Mat3 = SquareMatrix<real_type, 3>;
    //@}

  public:
    //! Construct with the new reference frame of the transformation
    explicit TransformTransformer(Transformation const& tr) : tr_{tr} {}

    //! Transform an identity
    Transformation const& operator()(NoTransformation const&) const
    {
        return tr_;
    }

    //!@{
    //! Transform a transformation
    inline Transformation operator()(Mat3 const&) const;
    inline Transformation operator()(Transformation const&) const;
    inline Transformation operator()(Translation const&) const;
    //!@}

  private:
    Transformation tr_;
};

//---------------------------------------------------------------------------//
/*!
 * Apply a transformation to a rotation matrix.
 */
Transformation TransformTransformer::operator()(Mat3 const& other) const
{
    return Transformation{gemm(tr_.rotation(), other), tr_.translation()};
}

//---------------------------------------------------------------------------//
/*!
 * Apply a transformation to a transform.
 */
Transformation
TransformTransformer::operator()(Transformation const& other) const
{
    return Transformation{gemm(tr_.rotation(), other.rotation()),
                          tr_.transform_up(other.translation())};
}

//---------------------------------------------------------------------------//
/*!
 * Apply a transformation to a translation.
 */
Transformation TransformTransformer::operator()(Translation const& other) const
{
    return Transformation{tr_.rotation(),
                          tr_.transform_up(other.translation())};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

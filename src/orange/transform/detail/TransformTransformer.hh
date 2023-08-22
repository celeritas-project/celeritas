//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/detail/TransformTransformer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "orange/MatrixUtils.hh"
#include "orange/Types.hh"

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
 * The operation returns a new transform defined
 * \f[
   \mathbf{T}' = \mathbf{T}_L \mathbf{T}_R
 * \f]
 * where T is the argument, \f$\mathbf{T}_L\f$ is the constructor argument (the
 * operator itself), and the result \f$\mathbf{T}'\f$ is the return value.
 * The resulting transform is:
 * \f[
   \mathbf{x}' = \mathbf{T}_L\mathbf{x}
   = \mathbf{R}_L\mathbf{x} + \mathbf{t}_L
 * \f]
 * Then
 * \f[
   \mathbf{T}_L\mathbf{T}_R\mathbf{x} = \mathbf{T}_L\mathbf{x}' =
   \mathbf{R}_L\mathbf{x}' + \mathbf{t}_L = \mathbf{R}_L(
   \mathbf{R}_R\mathbf{x} + \mathbf{t}_R) + \mathbf{t}_L
   \f]
 * Then the new transform has rotation
 * \f[
   \mathbf{R}' = \mathbf{R}_L\mathbf{R}_R
 * \f]
 * and translation
 * \f[
   \mathbf{t}' = \mathbf{R}_L\mathbf{t}_R + \mathbf{t}_L
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
    Transformation operator()(std::monostate) const { return tr_; }

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

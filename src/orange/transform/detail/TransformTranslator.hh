//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/detail/TransformTranslator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "corecel/math/ArrayOperators.hh"
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
 * Apply a translation to a transform to get another transform.
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
 * The resulting transform has rotation
 * \f[
   \mathbf{R}' = \mathbf{R}_R
 * \f]
 * and translation
 * \f[
   \mathbf{t}' = \mathbf{R}_L\mathbf{t}_R + \mathbf{t}_L
 * \f]
 */
class TransformTranslator
{
  public:
    //@{
    //! \name Type aliases
    using Mat3 = SquareMatrixReal3;
    //@}

  public:
    //! Construct with the new reference frame of the translation
    explicit TransformTranslator(Translation const& tr) : tr_{tr} {}

    //// TRANSFORMATIONS ////

    //! Transform an identity
    Translation operator()(std::monostate) const { return tr_; }

    inline Transformation operator()(Mat3 const&) const;

    inline Transformation operator()(Transformation const&) const;

    inline Translation operator()(Translation const&) const;

  private:
    Translation tr_;
};

//---------------------------------------------------------------------------//
/*!
 * Apply a translation to a rotation matrix.
 */
Transformation TransformTranslator::operator()(Mat3 const& rot) const
{
    return Transformation{rot, tr_.translation()};
}

//---------------------------------------------------------------------------//
/*!
 * Apply a translation to a transform.
 */
Transformation TransformTranslator::operator()(Transformation const& tr) const
{
    return Transformation{tr.rotation(), tr.translation() + tr_.translation()};
}

//---------------------------------------------------------------------------//
/*!
 * Apply a translation to another translation.
 */
Translation TransformTranslator::operator()(Translation const& tl) const
{
    return Translation{tl.translation() + tr_.translation()};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

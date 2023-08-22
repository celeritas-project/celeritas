//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTranslator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"

#include "Translation.hh"

namespace celeritas
{
class Transformation;
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

    Transformation operator()(Mat3 const&) const;

    Transformation operator()(Transformation const&) const;

    Translation operator()(Translation const&) const;

  private:
    Translation tr_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

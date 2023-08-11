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
 * This accepts a daughter-to-parent translation: the translation needed to
 * relocate a "lower" universe to a new coordinate system in the "higher"
 * universe.
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

//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTransformer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"

#include "Transformation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a transformation to another transformation.
 *
 * This accepts a daughter-to-parent transformation: the transformation needed
 * to relocate a "lower" universe to a new coordinate system in the "higher"
 * universe.
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

    //!@{
    //! Transform a transformation
    Transformation operator()(Mat3 const&) const;
    Transformation operator()(Transformation const&) const;
    Transformation operator()(Translation const&) const;
    //!@}

  private:
    Transformation tr_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

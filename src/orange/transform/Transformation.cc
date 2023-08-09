//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/Transformation.cc
//---------------------------------------------------------------------------//
#include "Transformation.hh"

#include "Translation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct as an identity transform.
 */
Transformation::Transformation() : Transformation{Translation{}} {}

//---------------------------------------------------------------------------//
/*!
 * Construct and check the input.
 */
Transformation::Transformation(Mat3 const& rot, Real3 const& trans)
    : rot_(rot), tra_(trans)
{
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
}  // namespace celeritas

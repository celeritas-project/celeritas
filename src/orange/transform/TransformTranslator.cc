//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTranslator.cc
//---------------------------------------------------------------------------//
#include "TransformTranslator.hh"

#include "corecel/math/ArrayOperators.hh"
#include "orange/MatrixUtils.hh"

#include "Transformation.hh"

namespace celeritas
{
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
}  // namespace celeritas

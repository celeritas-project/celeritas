//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTransformer.cc
//---------------------------------------------------------------------------//
#include "TransformTransformer.hh"

#include "orange/MatrixUtils.hh"
#include "orange/Types.hh"

#include "Transformation.hh"
#include "Translation.hh"

namespace celeritas
{
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
}  // namespace celeritas

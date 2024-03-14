//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformSimplifier.cc
//---------------------------------------------------------------------------//
#include "TransformSimplifier.hh"

#include "corecel/math/ArrayUtils.hh"
#include "orange/MatrixUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Translation may simplify to no transformation.
 */
VariantTransform TransformSimplifier::operator()(Translation const& t)
{
    if (norm(t.translation()) <= eps_)
    {
        // Distance to origin is less than tolerance
        return NoTransformation{};
    }
    return t;
}

//---------------------------------------------------------------------------//
/*!
 * Simplify, possibly to translation or no transform.
 *
 * See the derivation in the class documentation.
 */
VariantTransform TransformSimplifier::operator()(Transformation const& t)
{
    real_type tr = trace(t.rotation());
    if (tr >= 3 - ipow<2>(eps_))
    {
        // Rotation results in no more then epsilon movement
        return (*this)(Translation{t.translation()});
    }
    return t;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

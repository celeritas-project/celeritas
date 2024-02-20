//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformHasher.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>

#include "VariantTransform.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate a hash value of a transform for deduplication.
 */
class TransformHasher
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = std::size_t;
    //!@}

  public:
    // By default, calculate a hash based on the stored data
    template<class T>
    result_type operator()(T const&) const;

    // Special hash for "no transformation"
    result_type operator()(NoTransformation const&) const;

    // Special hash for "signed permutation"
    result_type operator()(SignedPermutation const&) const;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Calculate a hash for a variant transform
TransformHasher::result_type
visit(TransformHasher const&, VariantTransform const& transform);

//---------------------------------------------------------------------------//
}  // namespace celeritas

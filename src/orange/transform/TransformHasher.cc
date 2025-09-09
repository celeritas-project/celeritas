//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformHasher.cc
//---------------------------------------------------------------------------//
#include "TransformHasher.hh"

#include <functional>

#include "corecel/Assert.hh"
#include "corecel/math/HashUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * By default, calculate a hash based on the stored data.
 */
template<class T>
auto TransformHasher::operator()(T const& t) const -> result_type
{
    return hash_as_bytes(t.data());
}

//---------------------------------------------------------------------------//
/*!
 * Special hash for "no transformation".
 */
auto TransformHasher::operator()(NoTransformation const&) const -> result_type
{
    return std::hash<result_type>{}(0);
}

//---------------------------------------------------------------------------//
/*!
 * Special hash for "signed permutation".
 */
auto TransformHasher::operator()(SignedPermutation const& t) const
    -> result_type
{
    auto v = t.value();
    return std::hash<decltype(v)>{}(v);
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate a hash for a variant transform.
 */
TransformHasher::result_type
visit(TransformHasher const& th, VariantTransform const& transform)
{
    CELER_ASSUME(!transform.valueless_by_exception());
    return std::visit(th, transform);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
//! \cond

template std::size_t TransformHasher::operator()(Translation const&) const;
template std::size_t TransformHasher::operator()(Transformation const&) const;

//! \endcond
//---------------------------------------------------------------------------//
}  // namespace celeritas

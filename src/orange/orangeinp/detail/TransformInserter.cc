//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/TransformInserter.cc
//---------------------------------------------------------------------------//
#include "TransformInserter.hh"

#include "orange/transform/TransformHasher.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a pointer to the transform vector.
 */
TransformInserter::TransformInserter(VecTransform* transforms)
    : transform_{transforms}
    , cache_{0, HashTransform{transforms}, EqualTransform{transforms}}
{
    CELER_EXPECT(transform_);
    CELER_EXPECT(transform_->empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a transform with deduplication.
 */
TransformId TransformInserter::operator()(VariantTransform&& vt)
{
    CELER_ASSUME(!vt.valueless_by_exception());
    TransformId result = this->size_id();
    transform_->push_back(std::move(vt));
    auto [iter, inserted] = cache_.insert(result);
    if (!inserted)
    {
        // Roll back the change by erasing the last element
        transform_->pop_back();
        // Return the existing ID
        return *iter;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the hash of a transform.
 */
std::size_t TransformInserter::HashTransform::operator()(TransformId id) const
{
    CELER_EXPECT(storage && id < storage->size());
    return visit(TransformHasher{}, (*storage)[id.unchecked_get()]);
}

//---------------------------------------------------------------------------//
/*!
 * Compare two transform IDs for equality in a common container.
 */
bool TransformInserter::EqualTransform::operator()(TransformId a,
                                                   TransformId b) const
{
    CELER_EXPECT(storage && a < storage->size() && b < storage->size());
    return (*storage)[a.unchecked_get()] == (*storage)[b.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas

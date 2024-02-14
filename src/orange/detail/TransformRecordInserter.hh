//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/TransformRecordInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"

#include "../OrangeData.hh"
#include "../transform/VariantTransform.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a compressed transform from a variant.
 *
 * TODO: deduplicate transforms via hashes? Define special compressed transform
 * for 90-degree rotations?
 */
class TransformRecordInserter
{
  public:
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct with pointers to target data
    inline TransformRecordInserter(Items<TransformRecord>* transforms,
                                   Items<real_type>* reals);

    // Return a transform ID from a transform variant
    inline TransformId operator()(VariantTransform const& tr);

    // Construct a transform using known type
    template<class T>
    inline TransformId operator()(T const& tr);

  private:
    TransformId null_transform_;
    CollectionBuilder<TransformRecord> transforms_;
    DedupeCollectionBuilder<real_type> reals_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with pointers to target data.
 */
TransformRecordInserter::TransformRecordInserter(
    Items<TransformRecord>* transforms, Items<real_type>* reals)
    : transforms_{transforms}, reals_{reals}
{
    CELER_EXPECT(transforms && reals);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a transform variant.
 */
TransformId TransformRecordInserter::operator()(VariantTransform const& tr)
{
    CELER_ASSUME(!tr.valueless_by_exception());
    return std::visit(*this, tr);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a transform with a known type.
 */
template<class T>
TransformId TransformRecordInserter::operator()(T const& tr)
{
    // TODO: add equality and hash for TransformRecord and replace this with
    // just a dedupe collection builder
    if constexpr (std::is_same_v<T, NoTransformation>)
    {
        // Reuse the same null transform ID everywhere
        if (null_transform_)
        {
            return null_transform_;
        }
        // Save the transform ID for later
        null_transform_ = transforms_.size_id();
    }

    TransformRecord record;
    record.type = tr.transform_type();
    auto data = tr.data();
    record.data_offset = *reals_.insert_back(data.begin(), data.end()).begin();

    CELER_ASSERT(record);
    return transforms_.push_back(record);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

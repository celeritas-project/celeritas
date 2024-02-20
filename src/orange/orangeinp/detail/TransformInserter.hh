//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/TransformInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_set>
#include <vector>

#include "orange/transform/VariantTransform.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Deduplicate transforms as they're being built.
 *
 * This currently only works for *exact* transforms rather than *almost exact*
 * transforms. We may eventually want to add a "transform simplifier" and
 * "transform soft equal".
 */
class TransformInserter
{
  public:
    //!@{
    //! \name Type aliases
    using VecTransform = std::vector<VariantTransform>;
    //!@}

  public:
    // Construct with a pointer to the transform vector
    explicit TransformInserter(VecTransform* transforms);

    // Construct a transform with deduplication
    TransformId operator()(VariantTransform&& vt);

  private:
    //// TYPES ////

    struct HashTransform
    {
        VecTransform* storage{nullptr};
        std::size_t operator()(TransformId) const;
    };
    struct EqualTransform
    {
        VecTransform* storage{nullptr};
        bool operator()(TransformId, TransformId) const;
    };

    //// DATA ////

    VecTransform* transform_;
    std::unordered_set<TransformId, HashTransform, EqualTransform> cache_;

    //// HELPER FUNCTIONS ////

    //! Get the ID of the next transform to be inserted
    TransformId size_id() const { return TransformId(transform_->size()); }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas

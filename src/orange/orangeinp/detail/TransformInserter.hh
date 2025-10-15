//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * transforms.
 *
 * \todo Add "soft" transform comparisons (translation vectors should be
 * soft equal [magnitude of distance, compare difference between], and rotation
 * matrix times inverse (transpose) of other should result in a matrix that
 * satisfies \c soft_identity.
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

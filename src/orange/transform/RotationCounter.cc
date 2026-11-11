//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/RotationCounter.cc
//---------------------------------------------------------------------------//
#include "RotationCounter.hh"

#include <algorithm>
#include <variant>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/HashUtils.hh"
#include "orange/MatrixUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Hash a rotation matrix using its 9 values.
 */
std::size_t RotationCounter::RotationMatrixHasher::operator()(
    SquareMatrixReal3 const& mat) const
{
    return hash_as_bytes(Span<real_type const, 9>{&mat[0][0], 9});
}

//---------------------------------------------------------------------------//
/*!
 * Count a single transform's rotation.
 *
 * A transform without a rotation counts as the identity.
 */
void RotationCounter::count(VariantTransform const& transform)
{
    auto const* tr = std::get_if<Transformation>(&transform);
    ++counts_[tr ? tr->rotation() : make_identity()];
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the most common rotation and the number of transforms sharing it.
 *
 * If there is a tie, the rotation returned is whichever \c std::max_element
 * encounters first in the hash map's iteration order, which is not
 * guaranteed to be consistent across multiple platforms.
 */
auto RotationCounter::calc_most_common() const -> Results
{
    CELER_EXPECT(!counts_.empty());

    auto result = std::max_element(
        counts_.begin(), counts_.end(), [](auto const& a, auto const& b) {
            return a.second < b.second;
        });
    return {result->first, result->second};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

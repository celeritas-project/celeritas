//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/RotationCounter.cc
//---------------------------------------------------------------------------//
#include "RotationCounter.hh"

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
 * Transforms without a rotation are tallied separately rather than hashing an
 * identity matrix for each one, since they are typically the most common.
 */
void RotationCounter::count(VariantTransform const& transform)
{
    static SquareMatrixReal3 const identity = make_identity();

    auto const* tr = std::get_if<Transformation>(&transform);
    if (!tr || tr->rotation() == identity)
    {
        ++num_identity_;
        return;
    }
    ++counts_[tr->rotation()];
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the most common rotation and the number of transforms sharing it.
 *
 * The identity wins any tie. If there is a tie between other rotations, the
 * one returned is whichever is encountered first in the hash map's iteration
 * order, which is not guaranteed to be consistent across multiple platforms.
 * Therefore, client code should only use rotations when they constitute a
 * majority, i.e., > 50% of all transforms.
 */
auto RotationCounter::calc_most_common() const -> Results
{
    CELER_EXPECT(num_identity_ > 0 || !counts_.empty());

    Results result{make_identity(), num_identity_};
    for (auto const& [rot, count] : counts_)
    {
        if (count > result.count)
        {
            result = {rot, count};
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

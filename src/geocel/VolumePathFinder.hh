//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumePathFinder.hh
//! \sa test/geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"

#include "VolumeData.hh"
#include "VolumeView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reconstruct the volume-instance path for a \c VolumeUniqueInstanceId.
 *
 * Each \c VolumeUniqueInstanceId uniquely identifies a root-to-node path
 * (i.e., a Geant4 "touchable") in the volume DAG.  This class performs the
 * inverse of \c VolumePathAccumulator: given an ID it fills a
 * caller-supplied scratch buffer with the \c VolumeInstanceId sequence and
 * returns a (possibly shorter) span of the result.
 *
 * \c VolumeUniqueInstanceId{0} (or \c world_unique_instance ) always denotes
 * the world volume itself (empty path), and \c VolumeUniqueInstanceId{}
 * (null/invalid) is rejected with a precondition failure.
 * The valid range is \f$[0,\, N_\text{unique})\f$.
 *
 * The algorithm descends from the world volume level by level, always seeding
 * from the world's direct children.
 * At each level it scans the current volume's children to find the unique
 * child whose subtree contains the remaining UID, exploiting the fact that
 * sibling offsets are strictly increasing.
 * The cost is \f$O(D \log C)\f$ where \f$D\f$ is the path depth and \f$C\f$ is
 * the maximum number of children of any volume.
 *
 * The scratch buffer must be at least \c num_volume_levels - 1 long (the
 * maximum possible path depth).
 * Successive calls reuse the same buffer, so callers must consume the returned
 * span before the next call.
 *
 * \par Example:
 * \code
   std::vector<VolumeInstanceId> buf(params.num_volume_levels());
   VolumePathFinder find_path{params.host_ref(), make_span(buf)};

   VolumeUniqueInstanceId uid = ...;
   auto path = find_path(uid);   // [vi_0, vi_1, ..., vi_k]
 * \endcode
 */
class VolumePathFinder
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<VolumeParamsData>;
    using SpanVI = Span<VolumeInstanceId>;
    //!@}

  public:
    //! Construct from volume hierarchy data and a scratch buffer
    CELER_FUNCTION VolumePathFinder(ParamsRef const& params, SpanVI scratch)
        : params_(params), scratch_(scratch)
    {
        CELER_EXPECT(scratch_.size() + 1 >= params_.scalars.num_volume_levels);
    }

    // Reconstruct the path whose unique instance ID equals uid
    inline CELER_FUNCTION SpanVI operator()(VolumeUniqueInstanceId uid) const;

  private:
    ParamsRef const& params_;
    SpanVI scratch_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Reconstruct the path whose unique instance ID equals \c uid.
 *
 * Returns an empty span for UID 0 (the world volume, before any instance is
 * entered).  Otherwise writes the sequence of \c VolumeInstanceId values
 * from the world's first child down to the node identified by \c uid, and
 * returns a sub-span of the scratch buffer containing exactly those entries.
 */
CELER_FUNCTION auto
VolumePathFinder::operator()(VolumeUniqueInstanceId uid) const -> SpanVI
{
    CELER_EXPECT(uid < params_.scalars.num_unique_instances);
    using size_type = VolumeUniqueInstanceId::size_type;

    size_type offset = uid.unchecked_get();

    // Always seed from world's direct children
    auto parents = VolumeView{params_, params_.scalars.world}.children();
    VolumeLevelId::size_type depth = 0;

    while (offset > 0)
    {
        CELER_ASSERT(depth < scratch_.size());
        CELER_ASSERT(!parents.empty());

        // Sibling offsets are strictly increasing prefix sums; use
        // lower_bound to find the first child whose offset >= offset,
        // then step back one to get the chosen child. The offset of the first
        // parent is always greater than the current offset, so leave it out of
        // the search.
        auto comp = [this](VolumeInstanceId vi, size_type val) {
            return params_.unique_instance_offsets[vi] < val;
        };
        auto it = lower_bound(parents.begin() + 1, parents.end(), offset, comp);
        VolumeInstanceId chosen = *--it;

        scratch_[depth++] = chosen;
        offset -= params_.unique_instance_offsets[chosen] + size_type{1};
        parents = VolumeView{params_, params_.volume_ids[chosen]}.children();
    }

    return scratch_.first(depth);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

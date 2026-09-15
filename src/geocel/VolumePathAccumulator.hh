//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumePathAccumulator.hh
//! \sa test/geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "VolumeData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Incrementally compute the unique ID of a path in the volume hierarchy.
 *
 * Each \c VolumeUniqueInstanceId uniquely identifies a root-to-node path
 * (i.e., a Geant4 "touchable") in the volume DAG.  This class computes the ID
 * on the fly without allocating a path buffer: call \c operator() once for
 * each \c VolumeInstanceId encountered while descending from the world volume,
 * passing the current accumulated ID and receiving the updated one.
 *
 * For a volume instance \c vi at position \c k in its
 * parent volume's children list, the offset is the sum of
 * \c num_desc(volume(vj)) for all preceding siblings \c vj (positions
 * 0..k-1), where \c num_desc(V) counts the total number of unique paths
 * ending at any node in V's subtree (including V itself).
 * For a path \f$[vi_0, vi_1, \ldots, vi_k]\f$ the unique instance ID is
 * \f[
 *   \text{uid} = \sum_{i=0}^{k} \bigl(\text{offset}[vi_i] + 1\bigr).
 * \f]
 * The empty path (the world volume itself, with no enclosing instance) maps
 * to ID 0 (see \c world_unique_instance ).
 *
 * \par Example:
 * \code
   VolumePathAccumulator accum{params.host_ref()};
   VolumeUniqueInstanceId uid = world_unique_instance;
   for (VolumeInstanceId vi : path_below_world)
   {
       uid = accum(uid, vi);  // unique ID for the node reached via this step
   }
 * \endcode
 *
 * or, if it's easier to include the world as part of the path (using unsigned
 * integer overflow that's part of the OpaqueId implementation), or if the path
 * might be "empty" to indicate being outside:
 * \code
   VolumePathAccumulator accum{params.host_ref()};
   VolumeUniqueInstanceId uid;
   for (VolumeInstanceId vi : path_including_world)
   {
       uid = accum(uid, vi);  // unique ID for the node reached via this step
   }
 * \endcode

 *
 * \internal The mapping relies on \c unique_instance_offsets precomputed in
 * \c VolumeParamsData.
 */
class VolumePathAccumulator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<VolumeParamsData>;
    //!@}

  public:
    //! Construct from volume hierarchy data
    explicit CELER_FUNCTION VolumePathAccumulator(ParamsRef const& params)
        : params_(params)
    {
    }

    // Descend one level: add offset[vi]+1 to uid and return the result
    inline CELER_FUNCTION VolumeUniqueInstanceId operator()(
        VolumeUniqueInstanceId uid, VolumeInstanceId vi) const;

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Accumulate the contribution of one path step and return the updated ID.
 *
 * The result after \f$k\f$ calls equals the \c VolumeUniqueInstanceId of the
 * node reached via the first \f$k\f$ instances in the path.
 */
CELER_FUNCTION VolumeUniqueInstanceId VolumePathAccumulator::operator()(
    VolumeUniqueInstanceId uid, VolumeInstanceId vi) const
{
    CELER_EXPECT(vi < params_.unique_instance_offsets.size());
    auto const offset = params_.unique_instance_offsets[vi];
    return VolumeUniqueInstanceId(*uid + offset.get() + 1);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

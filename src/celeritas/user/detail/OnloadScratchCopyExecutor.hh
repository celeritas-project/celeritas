//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/OnloadScratchCopyExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/ThreadId.hh"

#include "../OnloadData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Equivalent to `CONT[tid]` but without debug checking.
 *
 * Debug checking causes the StepScratchCopyExecutor to grow large enough to
 * emit warnings.
 */
template<class C, class O>
CELER_FORCEINLINE_FUNCTION decltype(auto) xfast_get(C&& cont, OpaqueId<O> tid)
{
    static_assert(std::remove_reference_t<C>::memspace == MemSpace::native);
    return cont.data().get()[tid.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Compact entries, copying from a full state vector to one with # hits.
 *
 * The number of selected entries is read from device memory so that the
 * kernel can be launched over all track slots without first copying the
 * count to the host.
 */
struct OnloadScratchCopyExecutor
{
    NativeRef<OnloadStateData> state;
    ObserverPtr<size_type const> num_valid;

    // Gather results from selected tracks
    inline CELER_FUNCTION void operator()(ThreadId id);
};

//---------------------------------------------------------------------------//
/*!
 * Gather results from selected tracks.
 */
CELER_FUNCTION void OnloadScratchCopyExecutor::operator()(ThreadId dst_id)
{
    CELER_EXPECT(num_valid);
    if (!(dst_id.unchecked_get() < *num_valid))
    {
        // Past the end of the compacted data
        return;
    }
    CELER_EXPECT(state.size() == state.scratch.size()
                 && *num_valid <= state.size()
                 && dst_id < state.valid_id.size());

    // Indirect from thread to compressed track slot
    TrackSlotId src_id{xfast_get(state.valid_id, dst_id)};
    CELER_ASSERT(src_id < state.size());

#define DS_COPY_IF_SELECTED(FIELD) \
    do \
    { \
        if (!state.data.FIELD.empty()) \
        { \
            xfast_get(state.scratch.FIELD, dst_id) \
                = xfast_get(state.data.FIELD, src_id); \
        } \
    } while (0)

    DS_COPY_IF_SELECTED(energy);
    DS_COPY_IF_SELECTED(time);
    DS_COPY_IF_SELECTED(weight);
    DS_COPY_IF_SELECTED(pos);
    DS_COPY_IF_SELECTED(dir);
    DS_COPY_IF_SELECTED(track_id);
    DS_COPY_IF_SELECTED(particle_id);
#undef DS_COPY_IF_SELECTED
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

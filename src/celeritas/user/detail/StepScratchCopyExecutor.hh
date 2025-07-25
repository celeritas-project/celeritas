//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepScratchCopyExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ThreadId.hh"

#include "../StepData.hh"

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
CELER_FORCEINLINE_FUNCTION decltype(auto) fast_get(C&& cont, OpaqueId<O> tid)
{
    static_assert(std::remove_reference_t<C>::memspace == MemSpace::native);
    return cont.data().get()[tid.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Compact entries, copying from a full state vector to one with # hits.
 */
struct StepScratchCopyExecutor
{
    NativeRef<StepStateData> state;
    size_type num_valid{};

    // Gather results from active tracks that are in a detector
    inline CELER_FUNCTION void operator()(ThreadId id);
};

//---------------------------------------------------------------------------//
/*!
 * Gather results from active tracks that are in a detector.
 */
CELER_FUNCTION void StepScratchCopyExecutor::operator()(ThreadId dst_id)
{
    CELER_EXPECT(state.size() == state.scratch.size()
                 && num_valid <= state.size()
                 && dst_id < state.valid_id.size());

    // Indirect from thread to compressed track slot
    TrackSlotId src_id{fast_get(state.valid_id, dst_id)};
    CELER_ASSERT(src_id < state.size());

#define DS_COPY_IF_SELECTED(FIELD)                    \
    do                                                \
    {                                                 \
        if (!state.data.FIELD.empty())                \
        {                                             \
            fast_get(state.scratch.FIELD, dst_id)     \
                = fast_get(state.data.FIELD, src_id); \
        }                                             \
    } while (0)

    DS_COPY_IF_SELECTED(detector);
    DS_COPY_IF_SELECTED(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_COPY_IF_SELECTED(points[sp].time);
        DS_COPY_IF_SELECTED(points[sp].pos);
        DS_COPY_IF_SELECTED(points[sp].dir);
        DS_COPY_IF_SELECTED(points[sp].energy);

        if (auto const& data_vids = state.data.points[sp].volume_instance_ids;
            !data_vids.empty())
        {
            auto& scratch_vids = state.scratch.points[sp].volume_instance_ids;
            for (auto i : range(state.volume_instance_depth))
            {
                using ViId = ItemId<VolumeInstanceId>;

                ViId dst_vi_id{
                    dst_id.unchecked_get() * state.volume_instance_depth + i};
                ViId src_vi_id{
                    src_id.unchecked_get() * state.volume_instance_depth + i};
                fast_get(scratch_vids, dst_vi_id)
                    = fast_get(data_vids, src_vi_id);
            }
        }
    }

    DS_COPY_IF_SELECTED(event_id);
    DS_COPY_IF_SELECTED(parent_id);
    DS_COPY_IF_SELECTED(track_step_count);
    DS_COPY_IF_SELECTED(step_length);
    DS_COPY_IF_SELECTED(weight);
    DS_COPY_IF_SELECTED(particle);
    DS_COPY_IF_SELECTED(energy_deposition);
#undef DS_COPY_IF_SELECTED
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

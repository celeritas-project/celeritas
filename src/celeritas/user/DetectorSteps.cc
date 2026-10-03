//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cc
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "StepData.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::native>;

template<class T>
using ItemRef
    = celeritas::Collection<T, Ownership::reference, MemSpace::native>;

//---------------------------------------------------------------------------//
/*!
 * Whether the gathered data in a track slot is valid.
 *
 * If detectors are used, valid slots have a detector ID; otherwise valid
 * slots have a track ID.
 */
class IsSelected
{
  public:
    using StepDataRef
        = StepStateDataImpl<Ownership::reference, MemSpace::native>;

    explicit IsSelected(StepDataRef const& data) : data_{data} {}

    //! Number of track slots
    size_type size() const { return data_.size(); }

    //! Whether the slot is selected
    bool operator()(TrackSlotId tid) const
    {
        return data_.detector_id.empty()
                   ? static_cast<bool>(data_.track_id[tid])
                   : static_cast<bool>(data_.detector_id[tid]);
    }

  private:
    StepDataRef const& data_;
};

//---------------------------------------------------------------------------//
size_type count_num_valid(IsSelected const& is_selected)
{
    size_type size{0};
    for (TrackSlotId tid : range(TrackSlotId{is_selected.size()}))
    {
        if (is_selected(tid))
        {
            ++size;
        }
    }
    return size;
}

//---------------------------------------------------------------------------//
template<class T>
void assign_field(DetectorStepOutput::PinnedVec<T>* dst,
                  StateRef<T> const& src,
                  IsSelected const& is_selected,
                  size_type size)

{
    if (src.empty())
    {
        // This attribute is not in use
        dst->clear();
        return;
    }

    // Copy all items from valid threads
    dst->resize(size);

    auto iter = dst->begin();
    for (TrackSlotId tid : range(TrackSlotId{is_selected.size()}))
    {
        if (is_selected(tid))
        {
            *iter++ = src[tid];
        }
    }
    CELER_ASSERT(iter == dst->end());
}

//---------------------------------------------------------------------------//
template<class T>
void assign_field(DetectorStepOutput::PinnedVec<T>* dst,
                  ItemRef<T> const& src,
                  IsSelected const& is_selected,
                  size_type size,
                  size_type per_thread)

{
    if (src.empty())
    {
        // This attribute is not in use
        dst->clear();
        return;
    }

    // Copy all items from valid threads
    dst->resize(size * per_thread);

    auto iter = dst->begin();
    for (TrackSlotId tid : range(TrackSlotId{is_selected.size()}))
    {
        if (is_selected(tid))
        {
            for (size_type i = 0; i != per_thread; ++i)
            {
                *iter++ = src[ItemId<T>{per_thread * tid.unchecked_get() + i}];
            }
        }
    }
    CELER_ASSERT(iter == dst->end());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Consolidate results from selected tracks.
 *
 * Tracks are selected if they interacted with a detector or, if no detectors
 * are used, if their track ID was set during gathering.
 */
template<>
void copy_steps<MemSpace::host>(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::host> const& state)
{
    CELER_EXPECT(output);

    ScopedProfiling profile_this{"copy-steps"};

    // Get the number of threads that are active and selected
    IsSelected const is_selected{state.data};
    size_type size = count_num_valid(is_selected);

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    assign_field(&(output->FIELD), state.data.FIELD, is_selected, size)

    DS_ASSIGN(detector_id);
    DS_ASSIGN(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_ASSIGN(points[sp].time);
        DS_ASSIGN(points[sp].pos);
        DS_ASSIGN(points[sp].dir);
        DS_ASSIGN(points[sp].energy);
        if (state.num_volume_levels > 0)
        {
            assign_field(&(output->points[sp].volume_instance_ids),
                         state.data.points[sp].volume_instance_ids,
                         is_selected,
                         size,
                         state.num_volume_levels);
        }
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(primary_id);
    DS_ASSIGN(post_step_action_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(weight);
    DS_ASSIGN(particle_id);
    DS_ASSIGN(energy_deposition);
    DS_ASSIGN(hand_back_reason);

    output->num_volume_levels = state.num_volume_levels;

#undef DS_ASSIGN

    CELER_ENSURE(output->size() == size);
    CELER_ENSURE(output->track_id.size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

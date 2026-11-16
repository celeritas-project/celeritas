//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cc
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "StepData.hh"

#include "detail/StepCopyUtils.hh"

namespace celeritas
{
namespace
{
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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Consolidate results from selected tracks.
 *
 * Tracks are selected if they interacted with a detector or, if no detectors
 * are used, if their track ID was set during gathering. Device step data is
 * copied asynchronously with \c compact_steps_async and
 * \c copy_compacted_steps_async.
 */
void copy_steps(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::host> const& state)
{
    CELER_EXPECT(output);

    ScopedProfiling profile_this{"copy-steps"};

    // Get the number of threads that are active and selected
    IsSelected const is_selected{state.data};
    size_type size = detail::count_num_valid(is_selected);

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    detail::assign_field(&(output->FIELD), state.data.FIELD, is_selected, size)

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
            detail::assign_field(&(output->points[sp].volume_instance_ids),
                                 state.data.points[sp].volume_instance_ids,
                                 is_selected,
                                 size,
                                 state.num_volume_levels);
        }
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(generation);
    DS_ASSIGN(primary_id);
    DS_ASSIGN(post_step_action_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(weight);
    DS_ASSIGN(particle_id);
    DS_ASSIGN(energy_deposition);

    output->num_volume_levels = state.num_volume_levels;

#undef DS_ASSIGN

    CELER_ENSURE(output->size() == size);
    CELER_ENSURE(output->track_id.size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/OnloadSteps.cc
//---------------------------------------------------------------------------//
#include "OnloadSteps.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "OnloadData.hh"

#include "detail/StepCopyUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Whether the gathered data in a track slot is valid.
 *
 * Valid slots have a track ID.
 */
class IsSelected
{
  public:
    using StepDataRef
        = OnloadStateDataImpl<Ownership::reference, MemSpace::native>;

    explicit IsSelected(StepDataRef const& data) : data_{data} {}

    //! Number of track slots
    size_type size() const { return data_.size(); }

    //! Whether the slot is selected
    bool operator()(TrackSlotId tid) const
    {
        return static_cast<bool>(data_.track_id[tid]);
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
 * Tracks are selected if their track ID was set during gathering. Device step
 * data is copied asynchronously with \c compact_steps_async and
 * \c copy_compacted_steps_async.
 */
void copy_steps(
    OnloadStepOutput* output,
    OnloadStateData<Ownership::reference, MemSpace::host> const& state)
{
    CELER_EXPECT(output);

    ScopedProfiling profile_this{"copy-onload-steps"};

    // Get the number of threads that are active and selected
    IsSelected const is_selected{state.data};
    size_type size = detail::count_num_valid(is_selected);

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    detail::assign_field(&(output->FIELD), state.data.FIELD, is_selected, size)

    DS_ASSIGN(track_id);
    DS_ASSIGN(energy);
    DS_ASSIGN(time);
    DS_ASSIGN(pos);
    DS_ASSIGN(dir);

#undef DS_ASSIGN

    CELER_ENSURE(output->size() == size);
    CELER_ENSURE(output->track_id.size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

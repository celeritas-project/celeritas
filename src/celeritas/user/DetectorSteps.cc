//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cc
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"

#include "StepData.hh"

namespace celeritas
{
namespace
{
using DetectorRef
    = celeritas::StateCollection<DetectorId, Ownership::reference, MemSpace::host>;

template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::host>;

//---------------------------------------------------------------------------//
size_type count_num_valid(DetectorRef const& detector)
{
    size_type size{0};
    for (DetectorId id : detector[AllItems<DetectorId>{}])
    {
        if (id)
        {
            ++size;
        }
    }
    return size;
}

//---------------------------------------------------------------------------//
template<class T>
void assign_field(DetectorStepOutput::vector<T>* dst,
                  StateRef<T> const& src,
                  DetectorRef const& detector,
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
    for (TrackSlotId tid : range(TrackSlotId{src.size()}))
    {
        if (detector[tid])
        {
            *iter++ = src[tid];
        }
    }
    CELER_ASSERT(iter == dst->end());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Consolidate results from tracks that interacted with a detector.
 */
template<>
void copy_steps<MemSpace::host>(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::host> const& state)
{
    CELER_EXPECT(output);

    // Get the number of threads that are active and in a detector
    size_type size = count_num_valid(state.data.detector);

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    assign_field(&(output->FIELD), state.data.FIELD, state.data.detector, size)

    DS_ASSIGN(detector);
    DS_ASSIGN(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_ASSIGN(points[sp].time);
        DS_ASSIGN(points[sp].pos);
        DS_ASSIGN(points[sp].dir);
        DS_ASSIGN(points[sp].energy);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(particle);
    DS_ASSIGN(energy_deposition);
#undef DS_ASSIGN

    CELER_ENSURE(output->detector.size() == size);
    CELER_ENSURE(output->track_id.size() == size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

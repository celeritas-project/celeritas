//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct StepStateData;

//---------------------------------------------------------------------------//
/*!
 * CPU results for detector stepping at the beginning or end of a step.
 *
 * Since the volume has a one-to-one mapping to a DetectorId, we omit it. Since
 * multiple "touchable" (multi-level geometry instance) volumes can point to
 * the same detector, and the location in that hierarchy can be important, a
 * separate "volume instance IDs" multi-D level is available.
 */
struct DetectorStepPointOutput
{
    //// TYPES ////

    using Energy = units::MevEnergy;
    template<class T>
    using PinnedVec = std::vector<T, PinnedAllocator<T>>;

    //// DATA ////

    PinnedVec<real_type> time;
    PinnedVec<Real3> pos;
    PinnedVec<Real3> dir;
    PinnedVec<Energy> energy;

    PinnedVec<VolumeInstanceId> volume_instance_ids;
};

//---------------------------------------------------------------------------//
/*!
 * CPU results for many in-detector tracks at a single step iteration.
 *
 * This convenience class can be used to postprocess the results from sensitive
 * detectors on CPU. The data members will be available based on the \c
 * selection of the \c StepInterface class that gathered the data.
 *
 * Unlike \c StepStateData, which leaves gaps for inactive or filtered
 * tracks, every entry of these vectors will be valid. If detectors are
 * defined, each entry corresponds to a single DetectorId; otherwise
 * \c detector_id is empty and each entry corresponds to a selected track.
 * Entries are ordered by track slot.
 */
struct DetectorStepOutput
{
    //// TYPES ////

    using Energy = units::MevEnergy;
    template<class T>
    using PinnedVec = std::vector<T, PinnedAllocator<T>>;

    //// DATA ////

    // Pre- and post-step data
    EnumArray<StepPoint, DetectorStepPointOutput> points;

    // Track and detector IDs are always set
    PinnedVec<TrackId> track_id;
    PinnedVec<DetectorId> detector_id;

    // Additional optional data (sim)
    PinnedVec<EventId> event_id;
    PinnedVec<TrackId> parent_id;
    PinnedVec<PrimaryId> primary_id;
    PinnedVec<ActionId> post_step_action_id;
    PinnedVec<size_type> track_step_count;
    PinnedVec<real_type> step_length;
    PinnedVec<real_type> weight;

    // Additional optional data (physics)
    PinnedVec<ParticleId> particle_id;
    PinnedVec<Energy> energy_deposition;

    // 2D size for volume instances
    size_type num_volume_levels{0};

    //// METHODS ////

    //! Number of elements in the detector output.
    size_type size() const
    {
        return detector_id.empty() ? track_id.size() : detector_id.size();
    }
    //! Whether the size is nonzero
    explicit operator bool() const { return this->size() != 0; }
};

//---------------------------------------------------------------------------//
// Copy state data for all steps inside detectors to the output.
template<MemSpace M>
void copy_steps(DetectorStepOutput* output,
                StepStateData<Ownership::reference, M> const& state);

template<>
void copy_steps<MemSpace::host>(
    DetectorStepOutput*,
    StepStateData<Ownership::reference, MemSpace::host> const&);
template<>
void copy_steps<MemSpace::device>(
    DetectorStepOutput*,
    StepStateData<Ownership::reference, MemSpace::device> const&);

// Compact selected device step data without synchronizing the stream
void compact_steps_async(
    StepStateData<Ownership::reference, MemSpace::device> const& state,
    size_type* num_selected);

// Copy device step data compacted by compact_steps_async
void copy_compacted_steps(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::device> const& state,
    size_type num_selected);

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline void copy_steps<MemSpace::device>(
    DetectorStepOutput*,
    StepStateData<Ownership::reference, MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void compact_steps_async(
    StepStateData<Ownership::reference, MemSpace::device> const&, size_type*)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void copy_compacted_steps(
    DetectorStepOutput*,
    StepStateData<Ownership::reference, MemSpace::device> const&,
    size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas

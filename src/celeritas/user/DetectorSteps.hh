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
 * tracks, every entry of these vectors will be valid and correspond to a
 * single DetectorId.
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

    // Detector ID and track ID are always set
    PinnedVec<DetectorId> detector;
    PinnedVec<TrackId> track_id;

    // Additional optional data
    PinnedVec<EventId> event_id;
    PinnedVec<TrackId> parent_id;
    PinnedVec<PrimaryId> primary_id;
    PinnedVec<size_type> track_step_count;
    PinnedVec<real_type> step_length;
    PinnedVec<real_type> weight;
    PinnedVec<ParticleId> particle;
    PinnedVec<Energy> energy_deposition;

    // 2D size for volume instances
    size_type volume_instance_depth{0};

    //// METHODS ////

    //! Number of elements in the detector output.
    size_type size() const { return detector.size(); }
    //! Whether the size is nonzero
    explicit operator bool() const { return !detector.empty(); }
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

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline void copy_steps<MemSpace::device>(
    DetectorStepOutput*,
    StepStateData<Ownership::reference, MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas

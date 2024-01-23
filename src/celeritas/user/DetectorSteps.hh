//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 * Since the volume has a one-to-one mapping to a DetectorId, we omit it.
 */
struct DetectorStepPointOutput
{
    using Energy = units::MevEnergy;

    template<class T>
    using vector = std::vector<T, PinnedAllocator<T>>;

    vector<real_type> time;
    vector<Real3> pos;
    vector<Real3> dir;
    vector<Energy> energy;
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
    using Energy = units::MevEnergy;

    // Pre- and post-step data
    EnumArray<StepPoint, DetectorStepPointOutput> points;

    template<class T>
    using vector = std::vector<T, PinnedAllocator<T>>;

    // Detector ID and track ID are always set
    vector<DetectorId> detector;
    vector<TrackId> track_id;

    // Additional optional data
    vector<EventId> event_id;
    vector<TrackId> parent_id;
    vector<size_type> track_step_count;
    vector<real_type> step_length;
    vector<ParticleId> particle;
    vector<Energy> energy_deposition;

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

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/OnloadSteps.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/AsyncSnapshot.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class Stream;
template<Ownership W, MemSpace M>
struct OnloadStateData;

//---------------------------------------------------------------------------//
/*!
 * CPU results for many in-detector tracks at a single step iteration.
 *
 * Unlike \c OnloadStateData, which leaves gaps for inactive or filtered
 * tracks, every entry of these vectors will be valid. Entries are ordered by
 * track slot.
 */
struct OnloadStepOutput
{
    //// TYPES ////

    using Energy = units::MevEnergy;
    template<class T>
    using PinnedVec = std::vector<T, PinnedAllocator<T>>;

    //// DATA ////

    // Track and detector IDs are always set
    PinnedVec<TrackId> track_id;

    PinnedVec<Energy> energy;
    PinnedVec<real_type> time;
    PinnedVec<Real3> pos;
    PinnedVec<Real3> dir;

    //// METHODS ////

    //! Number of elements in the detector output.
    size_type size() const { return track_id.size(); }
    //! Whether the size is nonzero
    explicit operator bool() const { return this->size() != 0; }
};

//---------------------------------------------------------------------------//
// Copy host state data for all selected steps to the output
void copy_steps(
    OnloadStepOutput* output,
    OnloadStateData<Ownership::reference, MemSpace::host> const& state);

// Compact selected device step data without synchronizing the stream
void compact_steps_async(
    OnloadStateData<Ownership::reference, MemSpace::device> const& state,
    AsyncSnapshot<size_type>& num_selected);

// Enqueue a copy of device step data compacted by compact_steps_async
void copy_compacted_steps_async(
    OnloadStepOutput* output,
    OnloadStateData<Ownership::reference, MemSpace::device> const& state,
    size_type num_selected,
    Stream const& stream);

// Reserve output capacity for the largest possible compacted step
void reserve_compacted_steps(
    OnloadStepOutput* output,
    OnloadStateData<Ownership::reference, MemSpace::device> const& state);

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void reserve_compacted_steps(
    OnloadStepOutput*,
    OnloadStateData<Ownership::reference, MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void copy_compacted_steps_async(
    OnloadStepOutput*,
    OnloadStateData<Ownership::reference, MemSpace::device> const&,
    size_type,
    Stream const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void compact_steps_async(
    OnloadStateData<Ownership::reference, MemSpace::device> const&,
    AsyncSnapshot<size_type>&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas

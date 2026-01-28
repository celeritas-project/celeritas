//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/Collection.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * A single hit of a photon track on a sensitive detector.
 */
struct DetectorHit
{
    using Energy = units::MevEnergy;

    DetectorId detector{};
    Energy energy;
    real_type time;
    Real3 position;
    VolumeInstanceId volume_instance;
};

//---------------------------------------------------------------------------//
/*!
 * Pinned memory buffer for transferring detector hits.
 */
struct DetectorHitOutput
{
    //!@{
    //! \name Type aliases
    template<class T>
    using PinnedVec = std::vector<T, PinnedAllocator<T>>;
    //!@}

    PinnedVec<DetectorHit> hits;
};

//---------------------------------------------------------------------------//
/*!
 * State buffer for storing detector hits.
 *
 * Detector hits is large enough to store a hit for every track at the end of a
 * step. Stored hits may be invalid if the track is not in a detector region.
 */
template<Ownership W, MemSpace M>
struct DetectorStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    //!@}

    StateItems<DetectorHit> all_track_hits;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !all_track_hits.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return all_track_hits.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DetectorStateData<W, M>& operator=(DetectorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        all_track_hits = other.all_track_hits;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Copy hits from a memory space to pinned memory

template<MemSpace M>
void copy_hits(DetectorHitOutput* output,
               DetectorStateData<Ownership::reference, M> const& state,
               StreamId stream);

template<>
void copy_hits(DetectorHitOutput*,
               DetectorStateData<Ownership::reference, MemSpace::host> const&,
               StreamId);

template<>
void copy_hits(DetectorHitOutput*,
               DetectorStateData<Ownership::reference, MemSpace::device> const&,
               StreamId);

#if !CELER_USE_DEVICE
template<>
inline void
copy_hits(DetectorHitOutput*,
          DetectorStateData<Ownership::reference, MemSpace::device> const&,
          StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void
resize(DetectorStateData<Ownership::value, M>* state, size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    resize(&state->all_track_hits, size);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

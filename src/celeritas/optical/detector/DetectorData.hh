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
 */
struct DetectorHitOutput
{
    template<class T>
    using PinnedVec = std::vector<T, PinnedAllocator<T>>;

    PinnedVec<DetectorHit> hits;
};

//---------------------------------------------------------------------------//
/*!
 */
template<Ownership W, MemSpace M>
struct DetectorStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    //!@}

    StreamId stream_id{};
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
        stream_id = other.stream_id;
        all_track_hits = other.all_track_hits;
        return *this;
    }
};

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

template<MemSpace M>
void copy_hits(DetectorHitOutput* output,
               DetectorStateData<Ownership::reference, M> const& state);

template<>
void copy_hits(DetectorHitOutput* output,
               DetectorStateData<Ownership::reference, MemSpace::host> const&);

template<>
void copy_hits(DetectorHitOutput* output,
               DetectorStateData<Ownership::reference, MemSpace::device> const&);

#if !CELER_USE_DEVICE
template<>
inline void
copy_hits(DetectorHitOutput* output,
          DetectorStateData<Ownership::reference, MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

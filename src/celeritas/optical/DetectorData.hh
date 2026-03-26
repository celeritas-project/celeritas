//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/DetectorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/Collection.hh"
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
    real_type time{};
    Real3 position{};
    Real3 direction{};
    VolumeInstanceId volume_instance;

    //! An actual hit has a valid detector
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const
    {
        return static_cast<bool>(detector);
    }
};

//---------------------------------------------------------------------------//
/*!
 * State buffer for storing detector hits.
 *
 * Detector hits is large enough to store a hit for every track at the end of a
 * step. Stored hits may be invalid if the track is not in a detector region.
 *
 * When \c num_volume_levels is nonzero, \c volume_instance_ids stores the full
 * volume hierarchy for each track slot as a flat buffer of size
 * \c num_track_slots * num_volume_levels, indexed
 * \c [track_slot * num_volume_levels + level].
 */
template<Ownership W, MemSpace M>
struct DetectorStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    StateItems<DetectorHit> detector_hits;

    //! Flat volume hierarchy buffer: size = num_track_slots *
    //! num_volume_levels
    Items<VolumeInstanceId> volume_instance_ids;
    //! Number of volume levels per track slot (0 if hierarchy not stored)
    size_type num_volume_levels{0};

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !detector_hits.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return detector_hits.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DetectorStateData<W, M>& operator=(DetectorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        detector_hits = other.detector_hits;
        volume_instance_ids = other.volume_instance_ids;
        num_volume_levels = other.num_volume_levels;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 *
 * When \c num_levels is nonzero, allocates space for the full volume hierarchy
 * buffer (size * num_levels entries).
 */
template<MemSpace M>
inline void resize(DetectorStateData<Ownership::value, M>* state,
                   size_type size,
                   size_type num_levels = 0)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    resize(&state->detector_hits, size);

    if (num_levels > 0)
    {
        resize(&state->volume_instance_ids, size * num_levels);
        state->num_volume_levels = num_levels;
    }

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 * Host-side output of optical detector hits, including full volume hierarchy.
 *
 * This is populated by \c optical::DetectorAction and passed to the
 * \c HitCallbackFunc. The \c volume_instance_ids flat buffer is indexed as
 * \c [hit_idx * num_volume_levels + level], world at level 0, leaf at
 * \c num_volume_levels - 1.  The buffer is empty when \c num_volume_levels
 * is zero (i.e. Geant4 SD integration is disabled).
 */
struct DetectorHitsOutput
{
    std::vector<optical::DetectorHit> hits;
    //! Flat volume hierarchy for each hit (size = hits.size() *
    //! num_volume_levels)
    std::vector<VolumeInstanceId> volume_instance_ids;
    //! Number of volume levels per hit
    size_type num_volume_levels{0};

    //! True when there are hits to process
    explicit operator bool() const { return !hits.empty(); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

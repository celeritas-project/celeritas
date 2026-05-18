//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/DetectorData.hh
//---------------------------------------------------------------------------//
#pragma once

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
    PrimaryId primary{};
    Energy energy;
    real_type time{};
    Real3 position{};
    VolumeInstanceId volume_instance;
    VolumeUniqueInstanceId volume_unique_instance;

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

    //! Per track size of temporary storage for volume instance paths
    size_type scratch_path_size{0};

    StateItems<DetectorHit> detector_hits;
    Items<VolumeInstanceId> scratch_volume_path;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !detector_hits.empty() && !scratch_volume_path.empty()
               && scratch_path_size > 0;
    }

    //! State size
    CELER_FUNCTION size_type size() const { return detector_hits.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DetectorStateData<W, M>& operator=(DetectorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        detector_hits = other.detector_hits;
        scratch_volume_path = other.scratch_volume_path;
        scratch_path_size = other.scratch_path_size;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(DetectorStateData<Ownership::value, M>* state,
                   HostCRef<VolumeParamsData> const& volumes,
                   size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    // For GeoTrackInterface::volume_instance_id, the scratch span needs to be
    // at least one larger than the maximum geometry depth.
    state->scratch_path_size = volumes.scalars.num_volume_levels + 1;

    resize(&state->detector_hits, size);
    resize(&state->scratch_volume_path, size * state->scratch_path_size);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

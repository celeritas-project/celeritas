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
    Energy energy;
    real_type time{};
    Real3 position{};
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
 */
template<Ownership W, MemSpace M>
struct DetectorStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    //!@}

    StateItems<DetectorHit> detector_hits;

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
inline void
resize(DetectorStateData<Ownership::value, M>* state, size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    resize(&state->detector_hits, size);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

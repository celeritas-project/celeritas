//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

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

// //---------------------------------------------------------------------------//
// /*!
//  */
// struct DetectorHitOutput
// {
//     template<class T>
//     using PinnedVec = std::vector<T, PinnedAllocator<T>>;
//
//     PinnedVec<DetectorHit> hits;
// };
//
// //---------------------------------------------------------------------------//
// /*!
//  */
// template<Ownership W, MemSpace M>
// struct DetectorStateData
// {
//     template<class T>
//     using Items = StateCollection<T, W, M, TrackSlotId>;
//
//     StreamId stream_id{};
//     Items<DetectorHit> all_track_hits;
// };
//
// //---------------------------------------------------------------------------//
//
// template<MemSpace M>
// void copy_hits(DetectorHitOutput* output,
//                DetectorStateData<Ownership::reference, M> const& state);
//
// template<>
// void copy_hits(DetectorHitOutput* output,
//                DetectorStateData<Ownership::reference, MemSpace::host>
//                const&);
//
// template<>
// void copy_hits(DetectorHitOutput* output,
//                DetectorStateData<Ownership::reference, MemSpace::device>
//                const&);
//
// #if !CELER_USE_DEVICE
// template<>
// inline void
// copy_hits(DetectorHitOutput* output,
//           DetectorStateData<Ownership::reference, MemSpace::device> const&)
// {
//     CELER_NOT_CONFIGURED("CUDA OR HIP");
// }
// #endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

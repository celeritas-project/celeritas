//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/TrackInitAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
using TrackSlotRef = StateCollection<TrackSlotId, Ownership::reference, M>;
template<MemSpace M>
using TrackStatusRef = StateCollection<TrackStatus, Ownership::reference, M>;

//---------------------------------------------------------------------------//
//! Whether the track slot is vacant
struct IsVacant
{
    CELER_FUNCTION bool operator()(TrackStatus status) const
    {
        return status != TrackStatus::alive;
    }
};

//---------------------------------------------------------------------------//
// Compact the \c TrackSlotIds of the inactive tracks
size_type copy_if_vacant(TrackStatusRef<MemSpace::host> const&,
                         TrackSlotRef<MemSpace::host> const&,
                         StreamId);
size_type copy_if_vacant(TrackStatusRef<MemSpace::device> const&,
                         TrackSlotRef<MemSpace::device> const&,
                         StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline size_type copy_if_vacant(TrackStatusRef<MemSpace::device> const&,
                                TrackSlotRef<MemSpace::device> const&,
                                StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas

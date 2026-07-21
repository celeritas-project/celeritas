//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/WavelengthShiftData.hh"

#include "../GeneratorData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
using celeritas::optical::GeneratorDistributionData;
using celeritas::optical::WlsDistributionData;
using SPConstOpticalParams = std::shared_ptr<optical::CoreParams const>;

template<class T, MemSpace M>
using ItemsRef = Collection<T, Ownership::reference, M>;

//---------------------------------------------------------------------------//
// Remove all invalid distributions from the buffer.
template<class T>
size_type remove_if_invalid(
    ItemsRef<T, MemSpace::host> const&, size_type, size_type, StreamId);
template<class T>
size_type remove_if_invalid(
    ItemsRef<T, MemSpace::device> const&, size_type, size_type, StreamId);

//---------------------------------------------------------------------------//
// Count the number of optical photons in the distributions and add these to
// the number of pending  tracks.
void count_num_photons(
    SPConstOpticalParams params,
    optical::CoreState<MemSpace::host>&,
    ItemsRef<GeneratorDistributionData, MemSpace::host> const&,
    size_type,
    size_type,
    StreamId);
void count_num_photons(
    SPConstOpticalParams params,
    optical::CoreState<MemSpace::device>&,
    ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
    size_type,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<class T>
inline size_type remove_if_invalid(
    ItemsRef<T, MemSpace::device> const&, size_type, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

inline void
count_num_photons(SPConstOpticalParams,
                  optical::CoreState<MemSpace::device>&,
                  ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
                  size_type,
                  size_type,
                  StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

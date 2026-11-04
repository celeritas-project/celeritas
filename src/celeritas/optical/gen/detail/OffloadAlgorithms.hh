//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <string_view>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/StateDataStore.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedProfiling.hh"
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
void add_pending_photon_count(
    optical::CoreState<MemSpace::host>&,
    ItemsRef<GeneratorDistributionData, MemSpace::host> const&,
    size_type,
    size_type,
    StreamId);
void add_pending_photon_count(
    optical::CoreState<MemSpace::device>&,
    ItemsRef<GeneratorDistributionData, MemSpace::device> const&,
    size_type,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
// Grow a distribution buffer to hold at least this many distributions
template<template<Ownership, MemSpace> class S, MemSpace M>
inline void reserve_distributions(StateDataStore<S, M>* store,
                                  size_type num_used,
                                  size_type min_capacity,
                                  StreamId stream,
                                  std::string_view label);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Grow a distribution buffer to hold at least this many distributions.
 *
 * If the buffer is too small, it is reallocated with at least twice its
 * current capacity and the first \c num_used distributions and offsets are
 * copied into it. The store's reference is updated in place, but any copy of
 * it made before this call refers to the released storage and must not be used
 * afterward.
 *
 * \tparam S optical::GeneratorStateData or optical::WlsGeneratorStateData
 * \tparam M Memory space of the distribution buffer
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
void reserve_distributions(StateDataStore<S, M>* store,
                           size_type num_used,
                           size_type min_capacity,
                           StreamId stream,
                           std::string_view label)
{
    CELER_EXPECT(store);

    size_type const capacity = store->size();
    if (min_capacity <= capacity)
    {
        return;
    }

    size_type const new_capacity = std::max(min_capacity, 2 * capacity);
    CELER_LOG_LOCAL(info) << "Growing optical distribution buffer from "
                          << capacity << " to " << new_capacity << " in '"
                          << label << "'";

    ScopedProfiling profile_this{"reserve-distributions"};
    reserve(&store->value().distributions, num_used, new_capacity, stream);
    reserve(&store->value().offsets, num_used, new_capacity, stream);

    // Update the reference
    store->ref() = store->value();

    CELER_ENSURE(store->size() == new_capacity);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<class T>
inline size_type remove_if_invalid(
    ItemsRef<T, MemSpace::device> const&, size_type, size_type, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

inline void add_pending_photon_count(
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

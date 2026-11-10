//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"

#include "GeneratorCounterAlgorithms.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, MemSpace M>
using ItemsRef = Collection<T, Ownership::reference, M>;

//! Native pointer to a buffer size for use in class templates. Declared at
//! namespace scope because nvcc can fail to compile the unaliased type as a
//! member of a class template in unity builds.
using BufferSizePtr = ObserverPtr<size_type const, MemSpace::native>;

//---------------------------------------------------------------------------//
/*!
 * Find the index of the distribution from which to generate the primary.
 *
 * This finds the index in offsets for which offsets[result - 1] <= value <
 * offsets[result].
 */
inline CELER_FUNCTION size_type find_distribution_index(
    Span<size_type> offsets, size_type value)
{
    CELER_EXPECT(!offsets.empty());

    auto iter = celeritas::lower_bound(offsets.begin(), offsets.end(), value);
    CELER_ASSERT(iter != offsets.end());

    if (value == *iter)
    {
        ++iter;
    }
    return iter - offsets.begin();
}

//---------------------------------------------------------------------------//
/*!
 * Whether a distribution is valid.
 *
 * This returns a \c bool because distributions are only explicitly
 * convertible to one, and some device select algorithms assign the result of
 * the predicate to their flags.
 */
struct IsValid
{
    template<class T>
    CELER_FUNCTION bool operator()(T const& data) const
    {
        return static_cast<bool>(data);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Get the number of photons in a buffered distribution.
 *
 * Indices past the current buffer size, which is read from the state's
 * memory, contribute no photons. This allows fixed-size algorithms to be
 * launched over an upper bound of the buffer size.
 */
template<class T>
struct BufferedPhotonCount
{
    ObserverPtr<T const, MemSpace::native> data;
    BufferSizePtr size;

    CELER_FUNCTION size_type operator()(size_type i) const
    {
        return i < *size ? data[i].num_photons : 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Clear the buffered distributions past the current buffer size.
 */
template<class T>
struct ClearTailExecutor
{
    ObserverPtr<T, MemSpace::native> data;
    BufferSizePtr size;

    CELER_FUNCTION void operator()(ThreadId tid) const
    {
        if (!(tid.get() < *size))
        {
            data[tid.get()] = T{};
        }
    }
};

//---------------------------------------------------------------------------//
// Calculate the prefix sum of the photon counts and update the pending count
template<class T>
void inclusive_scan_photons(ItemsRef<T, MemSpace::host> const&,
                            ItemsRef<size_type, MemSpace::host> const&,
                            size_type,
                            GenCountersObserver<MemSpace::host>,
                            StreamId);
template<class T>
void inclusive_scan_photons(ItemsRef<T, MemSpace::device> const&,
                            ItemsRef<size_type, MemSpace::device> const&,
                            size_type,
                            GenCountersObserver<MemSpace::device>,
                            StreamId);

// Compact the valid distributions and update the buffer size
template<class T>
void select_valid(ItemsRef<T, MemSpace::host> const&,
                  size_type,
                  GenCountersObserver<MemSpace::host>,
                  StreamId);
template<class T>
void select_valid(ItemsRef<T, MemSpace::device> const&,
                  size_type,
                  GenCountersObserver<MemSpace::device>,
                  StreamId);

// Clear the distributions past the buffer size
template<class T>
void clear_tail(ItemsRef<T, MemSpace::host> const&,
                size_type,
                GenCountersObserver<MemSpace::host>,
                StreamId);
template<class T>
void clear_tail(ItemsRef<T, MemSpace::device> const&,
                size_type,
                GenCountersObserver<MemSpace::device>,
                StreamId);

#if !CELER_USE_DEVICE
template<class T>
inline void inclusive_scan_photons(ItemsRef<T, MemSpace::device> const&,
                                   ItemsRef<size_type, MemSpace::device> const&,
                                   size_type,
                                   GenCountersObserver<MemSpace::device>,
                                   StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

template<class T>
inline void select_valid(ItemsRef<T, MemSpace::device> const&,
                         size_type,
                         GenCountersObserver<MemSpace::device>,
                         StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

template<class T>
inline void clear_tail(ItemsRef<T, MemSpace::device> const&,
                       size_type,
                       GenCountersObserver<MemSpace::device>,
                       StreamId)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

}  // namespace detail
}  // namespace optical
}  // namespace celeritas

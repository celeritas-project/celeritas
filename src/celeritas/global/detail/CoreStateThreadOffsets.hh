//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/CoreStateThreadOffsets.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<MemSpace M>
class CoreStateThreadOffsets;

//---------------------------------------------------------------------------//
/*!
 * Holds Collections used by CoreState to store thread offsets.
 *
 * Note that \c ActionThreads is not "actions by thread" but is "threads by
 * action": it's indexed into using the action ID, and its value is the thread
 * ID at which the sorted state vector begins having an action.
 */
template<>
class CoreStateThreadOffsets<MemSpace::host>
{
  public:
    //!@{
    //! \name Type aliases
    using NativeActionThreads
        = Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>;
    using HostActionThreads = NativeActionThreads;
    //!@}

  public:
    auto& host_action_thread_offsets() { return thread_offsets_; }
    auto const& host_action_thread_offsets() const { return thread_offsets_; }
    auto& native_action_thread_offsets() { return thread_offsets_; }
    auto const& native_action_thread_offsets() const
    {
        return thread_offsets_;
    }

    //! Initialize using the number of actions
    void resize(size_type n) { celeritas::resize(&thread_offsets_, n); }

  private:
    NativeActionThreads thread_offsets_;
};

//---------------------------------------------------------------------------//
/*!
 * Holds Collections used by CoreState to store thread offsets.
 *
 * This is specialized for device memory space as two collections are needed,
 * one for the host and one for the device. Using pinned mapped memory would be
 * less efficient.
 */
template<>
class CoreStateThreadOffsets<MemSpace::device>
{
  public:
    //!@{
    //! \name Type aliases
    using NativeActionThreads
        = Collection<ThreadId, Ownership::value, MemSpace::device, ActionId>;
    using HostActionThreads
        = Collection<ThreadId, Ownership::value, MemSpace::mapped, ActionId>;
    //!@}

  public:
    auto& host_action_thread_offsets() { return host_thread_offsets_; }
    auto const& host_action_thread_offsets() const
    {
        return host_thread_offsets_;
    }
    auto& native_action_thread_offsets() { return thread_offsets_; }
    auto const& native_action_thread_offsets() const
    {
        return thread_offsets_;
    }
    void resize(size_type n)
    {
        celeritas::resize(&thread_offsets_, n);
        celeritas::resize(&host_thread_offsets_, n);
    }

  private:
    NativeActionThreads thread_offsets_;
    HostActionThreads host_thread_offsets_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

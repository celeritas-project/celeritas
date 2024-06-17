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
/*!
 * Holds Collections used by CoreState to store thread offsets.
 */
template<MemSpace M>
class CoreStateThreadOffsets
{
  public:
    //!@{
    //! \name Type aliases
    template<MemSpace M2>
    using ThreadActions = Collection<ThreadId, Ownership::value, M2, ActionId>;
    //!@}

  public:
    auto& host_action_thread_offsets() { return thread_offsets_; }
    auto const& host_action_thread_offsets() const { return thread_offsets_; }
    auto& native_action_thread_offsets()
    {
        return host_action_thread_offsets();
    }
    auto const& native_action_thread_offsets() const
    {
        return host_action_thread_offsets();
    }

    //! Initialize using the number of actions
    void resize(size_type n) { celeritas::resize(&thread_offsets_, n); }

  private:
    ThreadActions<M> thread_offsets_;
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
    template<MemSpace M>
    using ThreadActions = Collection<ThreadId, Ownership::value, M, ActionId>;
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
    ThreadActions<MemSpace::device> thread_offsets_;
    ThreadActions<MemSpace::mapped> host_thread_offsets_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

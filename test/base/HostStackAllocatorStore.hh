//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostStackAllocatorStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Span.hh"
#include "base/StackAllocatorInterface.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Debug helper class for storing stack allocations on the host.
 */
template<class T>
class HostStackAllocatorStore
{
    //!@{
    //! Type aliases
    using value_type      = T;
    using Pointers        = celeritas::StackAllocatorPointers<T>;
    using size_type       = typename Pointers::size_type;
    //!@}

  public:
    HostStackAllocatorStore() { this->resize(0); }

    // Resize and clear data.
    void resize(size_type capacity, value_type fill = {})
    {
        storage_.assign(capacity, fill);
        size_             = 0;
        pointers_.storage = celeritas::make_span(storage_);
        pointers_.size    = &size_;
    }

    //! Access allocated data
    celeritas::Span<const value_type> get() const
    {
        return {storage_.data(), size_};
    }

    //! Access host pointers
    const Pointers& host_pointers() const { return pointers_; }

  private:
    std::vector<value_type> storage_;
    size_type               size_;
    Pointers                pointers_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test

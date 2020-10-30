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
#include "base/StackAllocatorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host side version of StackAllocatorStore.
 */
template<class T>
class HostStackAllocatorStore
{
  public:
    //@{
    //! Type aliases
    using value_type      = T;
    using Pointers        = StackAllocatorPointers<T>;
    using size_type       = typename Pointers::size_type;
    using const_span_type = celeritas::span<const value_type>;
    //@}

  public:
    // Construct with defaults
    explicit HostStackAllocatorStore(size_type capacity)
        : storage_(capacity), size_(0)
    {
        pointers_.storage = make_span(storage_);
        pointers_.size    = &size_;
    }

    //! Size of the allocation
    size_type capacity() const { return storage_.size(); }

    //! Get the current size
    size_type get_size() { return size_; }

    //! Clear allocated data (as for StackAllocator, just sets size to 0)
    void clear() { size_ = 0; }

    /// HOST ACCESSORS ///

    // Get a view to the stack pointers
    Pointers host_pointers() { return pointers_; }

  private:
    std::vector<value_type> storage_;
    size_type               size_;
    Pointers                pointers_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

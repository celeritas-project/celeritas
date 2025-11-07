//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/MiniStack.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class that provides the functionality of a stack on an underlying
 * container.
 */
template<class T>
class MiniStack
{
  public:
    //!@{
    using size_type = ::celeritas::size_type;
    using value_type = T;
    //!@}

  public:
    //! Construct with underlying storage.
    CELER_FUNCTION explicit MiniStack(Span<T> storage)
        : data_(storage.data()), capacity_(storage.size())
    {
    }

    //! Insert a new element at the top of the stack
    CELER_FUNCTION void push(T element)
    {
        CELER_EXPECT(this->size() < this->capacity());
        data_[size_++] = element;
    }

    //! Remove and return the top element of the stack
    CELER_FUNCTION T pop()
    {
        CELER_EXPECT(!this->empty());
        return data_[--size_];
    }

    //! Whether there are any elements in the container
    CELER_FORCEINLINE_FUNCTION bool empty() const { return size_ == 0; }

    //! Get the number of elements
    CELER_FORCEINLINE_FUNCTION size_type size() const { return size_; }

    //! Get the number of elements that can fit in the allocated storage
    CELER_FORCEINLINE_FUNCTION size_type capacity() const { return capacity_; }

  private:
    T* data_;
    size_type size_{0};
    size_type capacity_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

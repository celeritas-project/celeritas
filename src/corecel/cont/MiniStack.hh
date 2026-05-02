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
 *
 * \par Example:
 * \code
    Array<int, 3> storage;
    MiniStack stack(Span{storage});

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(3, stack.capacity());

    // Push  and pop
    stack.push(42);
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(1, stack.size());
    EXPECT_EQ(42, stack.pop());
 * \endcode
 *
 */
template<class T, std::size_t Extent = dynamic_extent, class S = ::celeritas::size_type>
class MiniStack
{
  public:
    //!@{
    using value_type = T;
    using size_type = S;
    //!@}

  public:
    //! Construct with underlying storage.
    CELER_FUNCTION explicit MiniStack(Span<T, Extent> storage) : data_(storage)
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
    CELER_CEF bool empty() const { return size_ == 0; }

    //! Get the number of elements
    CELER_CEF size_type size() const { return size_; }

    //! Get the number of elements that can fit in the allocated storage
    CELER_CEF size_type capacity() const { return data_.size(); }

  private:
    Span<T, Extent> data_;
    size_type size_{0};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

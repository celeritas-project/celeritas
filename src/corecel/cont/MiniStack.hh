//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/MiniStack.hh
//! \sa corecel/cont/MiniStack.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Stack "view" that keeps the top element in a local scalar.
 *
 * The most recently pushed value is always stored in \c top_. Older values are
 * spilled into the underlying storage. This increases the effective stack
 * capacity by one: for storage size \c N, the stack capacity is \c N + 1.
 *
 * The internal \c size_ tracks only the number of spilled elements in
 * underlying storage. The logical size reported by \c size() is therefore
 * <code>size_ + !empty()</code>.
 *
 * \par Example:
 * \code
    Array<int, 3> storage;
    MiniStack stack(Span{storage});

    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(0, stack.size());
    EXPECT_EQ(4, stack.capacity());

    // Push  and pop
    stack.push(42);
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(1, stack.size());
    EXPECT_EQ(42, stack.top());
    stack.pop();
 * \endcode
 *
 */
template<class T, std::size_t Extent = dynamic_extent, class S = ::celeritas::size_type>
class MiniStack
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "MiniStack should be used only for trivial data");

  public:
    //!@{
    using value_type = T;
    using size_type = S;
    //!@}

  public:
    // Construct with underlying storage
    inline CELER_FUNCTION explicit MiniStack(Span<T, Extent> storage);

    // Insert a new element at the top of the stack
    inline CELER_FUNCTION void push(T element);

    // Remove the top element of the stack
    inline CELER_FUNCTION void pop();

    // Get the top element of the stack
    inline CELER_FUNCTION T top() const;

    //! Whether there are any elements in the container
    CELER_FIF bool empty() const { return empty_; }

    //! Get the number of elements
    CELER_FIF size_type size() const { return size_ + !this->empty(); }

    //! Get the number of elements that can fit in the allocated storage
    CELER_CEF size_type capacity() const { return spill_.size() + 1; }

  private:
    Span<T, Extent> spill_;
    T top_{};
    size_type size_{0};
    bool empty_{true};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with underlying spill storage.
 */
template<class T, std::size_t Extent, class S>
CELER_FUNCTION MiniStack<T, Extent, S>::MiniStack(Span<T, Extent> storage)
    : spill_(storage)
{
    CELER_ENSURE(this->empty());
}

//---------------------------------------------------------------------------//
/*!
 * Push an element onto the top of the stack.
 */
template<class T, std::size_t Extent, class S>
inline CELER_FUNCTION void MiniStack<T, Extent, S>::push(T element)
{
    CELER_EXPECT(this->size() < this->capacity());
    if (empty_)
    {
        empty_ = false;
    }
    else
    {
        spill_[size_++] = top_;
    }
    top_ = element;
}

//---------------------------------------------------------------------------//
/*!
 * Remove the top element.
 */
template<class T, std::size_t Extent, class S>
inline CELER_FUNCTION void MiniStack<T, Extent, S>::pop()
{
    CELER_EXPECT(!this->empty());
    if (size_ > 0)
    {
        top_ = spill_[--size_];
    }
    else
    {
        empty_ = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the top element.
 */
template<class T, std::size_t Extent, class S>
CELER_FIF T MiniStack<T, Extent, S>::top() const
{
    CELER_EXPECT(!this->empty());
    return top_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

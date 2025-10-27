//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/LogicStack.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Simple, fixed-max-size (no heap allocation) stack.
 *
 * This uses a bit field where the "top" of the stack is the least significant
 * bit.
 *
 * For a simple shape (the intersection of surfaces+senses), the max stack
 * depth is 1. For a combination of those, the stack depth is 2. This logic
 * stack can hold up to 32 entries (minimum size of size_type).
 *
 * The underlying code is highly optimized. For example, calling 'apply_and'
 * inside the 'evaluate' function (on GCC 5 with -O2) results in: \verbatim
        movl    %eax, %ecx  # stack, temp
        shrl    %eax        # D.44391
        andl    $1,   %ecx  #, temp
        andl    %ecx, %eax  # temp, stack
 * \endverbatim
 *
 * Furthermore, and delightfully, if LogicStack is local to a function, and
 * assertions are off, all operations on size_ are optimized out completely! So
 * there's no penalty to adding that extra safety check.
 */
class LogicStack
{
  public:
    //@{
    //! Typedefs
    using value_type = bool;
    using size_type = celeritas::size_type;
    //@}

  public:
    //! Default constructor
    CELER_FORCEINLINE_FUNCTION LogicStack() {}

    //! Greatest number of boolean values allowed on the stack
    static CELER_CONSTEXPR_FUNCTION size_type capacity()
    {
        return sizeof(size_type) * 8;
    }

    //// ACCESSORS ////

    //! Number of elements on the stack
    CELER_FORCEINLINE_FUNCTION size_type size() const { return size_; }

    // Whether any elements exist
    CELER_FORCEINLINE_FUNCTION bool empty() const;

    // Access the top value of the stack
    CELER_FORCEINLINE_FUNCTION value_type top() const;

    // Access a single bit (zero is deepest level of stack), used by ostream
    CELER_FORCEINLINE_FUNCTION value_type operator[](size_type index) const;

    //// MUTATORS ////

    // Push a boolean onto the stack
    CELER_FORCEINLINE_FUNCTION void push(value_type v);

    // Pop a value off the stack
    CELER_FORCEINLINE_FUNCTION value_type pop();

    // Negate the value on the top of the stack
    CELER_FORCEINLINE_FUNCTION void apply_not();

    // Apply boolean 'and' to the top of the stack
    CELER_FORCEINLINE_FUNCTION void apply_and();

    // Apply boolean 'or' to the top of the stack
    CELER_FORCEINLINE_FUNCTION void apply_or();

  private:
    //! Get the least significant bit
    static CELER_CONSTEXPR_FUNCTION size_type lsb(size_type val)
    {
        return val & size_type(1);
    }

    //! Shift right by one
    static CELER_CONSTEXPR_FUNCTION size_type shr(size_type val)
    {
        return val >> size_type(1);
    }

    //! Shift left by one
    static CELER_CONSTEXPR_FUNCTION size_type shl(size_type val)
    {
        return val << size_type(1);
    }

  private:
    //// DATA ////

    size_type data_{0};  //!< Stack data
    size_type size_{0};  //!< Stack depth
};

//---------------------------------------------------------------------------//
/*!
 * Whether the stack has any pushed values.
 */
CELER_FUNCTION bool LogicStack::empty() const
{
    return size_ == size_type(0);
}

//---------------------------------------------------------------------------//
/*!
 * Access the top value of the stack.
 */
CELER_FUNCTION auto LogicStack::top() const -> value_type
{
    CELER_EXPECT(!empty());
    return LogicStack::lsb(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Access a single bit (zero is deepest level of stack).
 */
CELER_FUNCTION auto LogicStack::operator[](size_type index) const -> value_type
{
    CELER_EXPECT(index < size());
    size_type shift = size() - index - size_type(1);
    return LogicStack::lsb(data_ >> shift);
}

//---------------------------------------------------------------------------//
/*!
 * Push a boolean onto the stack.
 */
CELER_FUNCTION void LogicStack::push(value_type v)
{
    CELER_EXPECT(size() != capacity());
    // Shift stack left and add least significant bit
    data_ = LogicStack::shl(data_) | LogicStack::lsb(v);
    // Size for DBC
    ++size_;
}

//---------------------------------------------------------------------------//
/*!
 * Pop a value off the stack.
 */
CELER_FUNCTION auto LogicStack::pop() -> value_type
{
    CELER_EXPECT(!empty());
    // Extract least significant bit
    value_type result = LogicStack::lsb(data_);
    // Shift right
    data_ = LogicStack::shr(data_);
    // Update size
    --size_;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Negate the value on the top of the stack.
 */
CELER_FUNCTION void LogicStack::apply_not()
{
    CELER_EXPECT(!empty());
    data_ ^= size_type(1);
}

//---------------------------------------------------------------------------//
/*!
 * Apply boolean 'and' to the top of the stack.
 */
CELER_FUNCTION void LogicStack::apply_and()
{
    CELER_EXPECT(size() >= size_type(2));
    size_type temp = LogicStack::lsb(data_);
    data_ = LogicStack::shr(data_) & (temp | ~size_type(1));
    --size_;
}

//---------------------------------------------------------------------------//
/*!
 * Apply boolean 'or' to the top of the stack.
 */
CELER_FUNCTION void LogicStack::apply_or()
{
    CELER_EXPECT(size() >= size_type(2));
    data_ = LogicStack::shr(data_) | LogicStack::lsb(data_);
    --size_;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

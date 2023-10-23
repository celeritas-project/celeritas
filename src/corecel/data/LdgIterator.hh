//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgIterator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 *
 */
template<class T>
class LDGIterator
{
  public:
    //!@{
    //! \name Type aliases
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<T>;
    using reference = std::add_const_t<std::add_lvalue_reference_t<T>>;
    using iterator_catgeory = std::random_access_iterator_tag;
    using self = LDGIterator<T>;
    //!@}

  public:
    //!@{
    //! Construct a pointer
    constexpr LDGIterator() noexcept = default;
    constexpr LDGIterator(LDGIterator<T> const&) noexcept = default;
    CELER_CONSTEXPR_FUNCTION LDGIterator(std::nullptr_t) noexcept {}
    CELER_CONSTEXPR_FUNCTION explicit LDGIterator(pointer ptr) noexcept
        : ptr_{ptr}
    {
    }
    //!@}

    // Iterator requirements
    CELER_CONSTEXPR_FUNCTION reference operator*() const
    {
        // TODO: __ldg
        return *ptr_;
    }
    CELER_CONSTEXPR_FUNCTION self& operator++()
    {
        ++ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self& operator=(self const& it)
    {
        if (this == &it)
        {
            return *this;
        }
        ptr_ = it.ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION void swap(self& it) noexcept
    {
        ::celeritas::trivial_swap(ptr_, it.ptr_);
    }

    // ForwardIterator requirements
    CELER_CONSTEXPR_FUNCTION self operator++(int)
    {
        self tmp{ptr_};
        ++ptr_;
        return tmp;
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator==(self const& lhs, self const& rhs)
    {
        return lhs.ptr_ == rhs.ptr_;
    }
    CELER_CONSTEXPR_FUNCTION pointer operator->() { return **this; }

    // BidirectionalIterator requirements
    CELER_CONSTEXPR_FUNCTION self& operator--()
    {
        --ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self operator--(int)
    {
        self tmp{ptr_};
        --ptr_;
        return tmp;
    }

    // RandomAccessIterator requirements
    CELER_CONSTEXPR_FUNCTION self& operator+=(const difference_type n)
    {
        ptr_ += n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self operator+(const difference_type n) const
    {
        self tmp{ptr_};
        return tmp += n;
    }
    CELER_CONSTEXPR_FUNCTION self& operator-=(const difference_type n)
    {
        ptr_ -= n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self operator-(const difference_type n) const
    {
        self tmp{ptr_};
        return tmp -= n;
    }
    CELER_CONSTEXPR_FUNCTION difference_type operator-(self const& it) const
    {
        return it.ptr_ - ptr_;
    }
    CELER_CONSTEXPR_FUNCTION reference operator[](const difference_type n) const
    {
        // TODO: __ldg
        return *(ptr_ + n);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator<(self const& lhs, self const& rhs)
    {
        return lhs.ptr_ < rhs.ptr_;
    }

  private:
    const pointer ptr_;
};

// RandomAccessIterator requirements
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator>(LDGIterator<T> const& lhs, LDGIterator<T> const& rhs)
{
    return rhs < lhs;
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator<=(LDGIterator<T> const& lhs, LDGIterator<T> const& rhs)
{
    return !(lhs > rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator>=(LDGIterator<T> const& lhs, LDGIterator<T> const& rhs)
{
    return !(lhs < rhs);
}
// ForwardIterator requirements
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(LDGIterator<T> const& lhs, LDGIterator<T> const& rhs)
{
    return !(lhs == rhs);
}

//! Swap two pointers
template<class T>
CELER_CONSTEXPR_FUNCTION void
swap(LDGIterator<T>& lhs, LDGIterator<T>& rhs) noexcept
{
    return lhs.swap(rhs);
}

template<class T>
inline LDGIterator<T> make_ldgiterator(T* ptr) noexcept
{
    return LDGIterator<T>{ptr};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
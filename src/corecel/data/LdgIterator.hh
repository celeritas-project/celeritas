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

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Iterator for read-only device data. Use __ldg intrinsic to load data in
 * read-only cache.
 */
template<class T>
class LDGIterator
{
  public:
    //!@{
    //! \name Type aliases
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<T const>;
    using reference = std::add_lvalue_reference_t<T const>;
    using iterator_catgeory = std::random_access_iterator_tag;
    using self = LDGIterator<T>;
    //!@}

  public:
    //!@{
    //! Construct a pointer
    constexpr LDGIterator() noexcept = default;
    constexpr LDGIterator(self const&) noexcept = default;
    CELER_CONSTEXPR_FUNCTION LDGIterator(std::nullptr_t) noexcept {}
    CELER_CONSTEXPR_FUNCTION explicit LDGIterator(pointer ptr) noexcept
        : ptr_{ptr}
    {
    }
    //!@}

    // Iterator requirements
    CELER_CONSTEXPR_FUNCTION reference operator*() const noexcept
    {
#if CELER_DEVICE_COMPILE
        return __ldg(ptr_);
#else
        return *ptr_;
#endif
    }
    CELER_CONSTEXPR_FUNCTION self& operator++() noexcept
    {
        ++ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self& operator=(self const& it) noexcept
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
    CELER_CONSTEXPR_FUNCTION self operator++(int) noexcept
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
    CELER_CONSTEXPR_FUNCTION pointer operator->() const noexcept
    {
        return ptr_;
    }

    // BidirectionalIterator requirements
    CELER_CONSTEXPR_FUNCTION self& operator--() noexcept
    {
        --ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self operator--(int) noexcept
    {
        self tmp{ptr_};
        --ptr_;
        return tmp;
    }

    // RandomAccessIterator requirements
    CELER_CONSTEXPR_FUNCTION self& operator+=(const difference_type n) noexcept
    {
        ptr_ += n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self operator+(const difference_type n) const noexcept
    {
        self tmp{ptr_};
        return tmp += n;
    }
    CELER_CONSTEXPR_FUNCTION self& operator-=(const difference_type n) noexcept
    {
        ptr_ -= n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION self operator-(const difference_type n) const noexcept
    {
        self tmp{ptr_};
        return tmp -= n;
    }
    CELER_CONSTEXPR_FUNCTION difference_type
    operator-(self const& it) const noexcept
    {
        return it.ptr_ - ptr_;
    }
    CELER_CONSTEXPR_FUNCTION reference
    operator[](const difference_type n) const noexcept
    {
#if CELER_DEVICE_COMPILE
        return __ldg(ptr_ + n);
#else
        return ptr_[n];
#endif
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator<(self const& lhs, self const& rhs)
    {
        return lhs.ptr_ < rhs.ptr_;
    }

    // Conversion operators
    CELER_CONSTEXPR_FUNCTION explicit operator pointer() const noexcept
    {
        return ptr_;
    }
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const noexcept
    {
        return ptr_ != nullptr;
    }

  private:
    pointer ptr_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

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
template<class T>
CELER_CONSTEXPR_FUNCTION LDGIterator<T>
operator+(const typename LDGIterator<T>::difference_type n,
          LDGIterator<T> const& it)
{
    return it + n;
}

// ForwardIterator requirements
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(LDGIterator<T> const& lhs, LDGIterator<T> const& rhs)
{
    return !(lhs == rhs);
}

// Iterator requirements
template<class T>
CELER_CONSTEXPR_FUNCTION void
swap(LDGIterator<T>& lhs, LDGIterator<T>& rhs) noexcept
{
    return lhs.swap(rhs);
}

// Helper

template<class T>
inline LDGIterator<T>
make_ldgiterator(typename LDGIterator<T>::pointer ptr) noexcept
{
    return LDGIterator<T>{ptr};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
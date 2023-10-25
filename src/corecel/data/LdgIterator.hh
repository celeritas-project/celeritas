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

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/detail/LdgIteratorImpl.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Iterator for read-only device data. Use __ldg intrinsic to load data in
 * read-only cache.
 */
template<class T>
class LdgIterator
{
    //!@{
    //! \name Type aliases
  private:
    using LDGLoadPolicy = detail::LdgLoader<T>;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = typename LDGLoadPolicy::value_type;
    using pointer = typename LDGLoadPolicy::pointer;
    using reference = typename LDGLoadPolicy::reference;
    using iterator_category = std::random_access_iterator_tag;
    //!@}

  public:
    //!@{
    //! Construct a pointer
    constexpr LdgIterator() noexcept = default;
    constexpr LdgIterator(LdgIterator const&) noexcept = default;
    CELER_CONSTEXPR_FUNCTION LdgIterator(std::nullptr_t) noexcept {}
    CELER_CONSTEXPR_FUNCTION explicit LdgIterator(pointer ptr) noexcept
        : ptr_{ptr}
    {
    }
    //!@}

    //!@{
    //! \name RandomAccessIterator requirements
    CELER_CONSTEXPR_FUNCTION reference operator*() const noexcept
    {
        return LDGLoadPolicy::read(ptr_);
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator& operator++() noexcept
    {
        ++ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION void swap(LdgIterator& it) noexcept
    {
        ::celeritas::trivial_swap(ptr_, it.ptr_);
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator operator++(int) noexcept
    {
        LdgIterator tmp{ptr_};
        ++ptr_;
        return tmp;
    }
    CELER_CONSTEXPR_FUNCTION pointer operator->() const noexcept
    {
        return ptr_;
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator& operator--() noexcept
    {
        --ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator operator--(int) noexcept
    {
        LdgIterator tmp{ptr_};
        --ptr_;
        return tmp;
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator&
    operator+=(const difference_type n) noexcept
    {
        ptr_ += n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator&
    operator-=(const difference_type n) noexcept
    {
        ptr_ -= n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION reference
    operator[](const difference_type n) const noexcept
    {
        return LDGLoadPolicy::read(ptr_ + n);
    }
    //!@}

    // Conversion operators
    //!@{
    //! \name Conversion operators
    CELER_CONSTEXPR_FUNCTION explicit operator pointer() const noexcept
    {
        return ptr_;
    }
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const noexcept
    {
        return ptr_ != nullptr;
    }
    //!@}

  private:
    pointer ptr_{nullptr};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

//!@{
//! RandomAccessIterator requirements
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator==(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept
{
    using pointer = typename LdgIterator<T>::pointer;
    return static_cast<pointer>(lhs) == static_cast<pointer>(rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept
{
    return !(lhs == rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator<(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept
{
    using pointer = typename LdgIterator<T>::pointer;
    return static_cast<pointer>(lhs) < static_cast<pointer>(rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator>(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept
{
    return rhs < lhs;
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator<=(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept
{
    return !(lhs > rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator>=(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept
{
    return !(lhs < rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION LdgIterator<T>
operator+(LdgIterator<T> const& it,
          const typename LdgIterator<T>::difference_type n) noexcept
{
    LdgIterator tmp{it};
    return tmp += n;
}
template<class T>
CELER_CONSTEXPR_FUNCTION LdgIterator<T>
operator+(const typename LdgIterator<T>::difference_type n,
          LdgIterator<T> const& it) noexcept
{
    return it + n;
}
template<class T>
CELER_CONSTEXPR_FUNCTION LdgIterator<T>
operator-(LdgIterator<T> const& it,
          const typename LdgIterator<T>::difference_type n) noexcept
{
    LdgIterator<T> tmp{it};
    return tmp -= n;
}
template<class T>
CELER_CONSTEXPR_FUNCTION auto
operator-(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs) noexcept ->
    typename LdgIterator<T>::difference_type
{
    using pointer = typename LdgIterator<T>::pointer;
    return static_cast<pointer>(lhs) - static_cast<pointer>(rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION void
swap(LdgIterator<T>& lhs, LdgIterator<T>& rhs) noexcept
{
    return lhs.swap(rhs);
}
//!@}

//!@{
//! Helper
template<class T>
inline LdgIterator<T> make_LdgIterator(T const* ptr) noexcept
{
    return LdgIterator<T>{ptr};
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
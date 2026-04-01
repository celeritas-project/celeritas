//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/detail/LdgIterator.hh
//! \sa corecel/data/Ldg.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <iterator>
#include <type_traits>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
class LdgRefWrapper;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Proxy iterator that constructs a LdgRefWrapper when dereferenced.
 * \tparam T value type being accessed
 *
 * See LdgRefWrapper and \c ldg .
 */
template<class T>
class LdgIterator
{
    static_assert(std::is_const_v<T>);

  public:
    //!@{
    //! \name Type aliases
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_const_t<T>;
    using pointer = T*;
    using reference = LdgRefWrapper<T>;
    using iterator_category = std::random_access_iterator_tag;
    //!@}

  public:
    //!@{
    //! Construct a pointer
    constexpr LdgIterator() noexcept = default;
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
        return LdgRefWrapper<T>{*ptr_};
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator& operator++() noexcept
    {
        ++ptr_;
        return *this;
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
    CELER_CONSTEXPR_FUNCTION LdgIterator& operator+=(difference_type n) noexcept
    {
        ptr_ += n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION LdgIterator& operator-=(difference_type n) noexcept
    {
        ptr_ -= n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION reference operator[](difference_type n) const noexcept
    {
        return LdgRefWrapper<T>{*(ptr_ + n)};
    }
    //!@}

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

    //!@{
    //! RandomAccessIterator requirements
    CELER_CONSTEXPR_FUNCTION friend bool
    operator==(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return static_cast<pointer>(lhs) == static_cast<pointer>(rhs);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator!=(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return !(lhs == rhs);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator==(LdgIterator const& it, std::nullptr_t) noexcept
    {
        return !static_cast<bool>(it);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator!=(LdgIterator const& it, std::nullptr_t) noexcept
    {
        return static_cast<bool>(it);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator==(std::nullptr_t, LdgIterator const& it) noexcept
    {
        return !static_cast<bool>(it);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator!=(std::nullptr_t, LdgIterator const& it) noexcept
    {
        return static_cast<bool>(it);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator<(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return static_cast<pointer>(lhs) < static_cast<pointer>(rhs);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator>(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return rhs < lhs;
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator<=(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return !(lhs > rhs);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator>=(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return !(lhs < rhs);
    }
    CELER_CONSTEXPR_FUNCTION friend LdgIterator
    operator+(LdgIterator const& it, difference_type const n) noexcept
    {
        return LdgIterator{it} += n;
    }
    CELER_CONSTEXPR_FUNCTION friend LdgIterator
    operator+(difference_type const n, LdgIterator const& it) noexcept
    {
        return it + n;
    }
    CELER_CONSTEXPR_FUNCTION friend LdgIterator
    operator-(LdgIterator const& it, difference_type const n) noexcept
    {
        return LdgIterator{it} -= n;
    }
    CELER_CONSTEXPR_FUNCTION friend difference_type
    operator-(LdgIterator const& lhs, LdgIterator const& rhs) noexcept
    {
        return static_cast<pointer>(lhs) - static_cast<pointer>(rhs);
    }
    //!@}

  private:
    pointer ptr_{nullptr};
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class T>
LdgIterator(T*) -> LdgIterator<std::add_const_t<T>>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

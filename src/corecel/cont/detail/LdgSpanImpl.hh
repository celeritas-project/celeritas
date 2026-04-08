//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/detail/LdgSpanImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "corecel/Macros.hh"
#include "corecel/data/Ldg.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, class = void>
struct IsLdgSupported : std::false_type
{
    static_assert(std::is_const_v<T>);
};

template<class T>
struct IsLdgSupported<T, std::void_t<decltype(ldg_data(std::declval<T*>()))>>
    : std::true_type
{
};

//! Whether a type is supported by \c ldg
template<class T>
inline constexpr bool is_ldg_supported_v = IsLdgSupported<T>::value;

//---------------------------------------------------------------------------//
/*!
 * Wrapper that loads data safely via \c ldg on conversion.
 * \tparam T value type (must be const) being accessed
 *
 * This class mirrors \c std::reference_wrapper, storing a pointer to a const
 * object and provides an implicit conversion to the value type.
 * However, the value being wrapped \em must be a const reference, and the
 * return is a \c value rather than a reference .
 *
 * The \c __ldg intrinsic is invoked during the implicit load, so client code
 * can bind the wrapper to an ordinary value variable transparently.
 */
template<class T>
class LdgWrapper
{
    static_assert(std::is_const_v<T>);
    static_assert(is_ldg_supported_v<T>, "type is incompatible with ldg");

  public:
    //!@{
    //! \name Type aliases
    using type = std::remove_const_t<T>;
    //!@}

  public:
    //! Construct from a const reference to the target
    CELER_CEF LdgWrapper(T& ref) noexcept : ptr_{&ref} {}

    //! Load the referenced value using __ldg
    CELER_CEF type get() const noexcept { return ldg(ptr_); }

    //! Implicit conversion: load via __ldg
    CELER_CEF operator type() const noexcept { return this->get(); }

    //!@{
    /*!
     * Comparison operators against the underlying type or another wrapper.
     *
     * Defined here so template \c operator== (e.g. \c OpaqueId) are found via
     * ADL without requiring implicit conversion during deduction.
     * The \c LdgWrapper vs \c LdgWrapper overloads prevent ambiguity when
     * both arguments are wrappers (e.g., \c std::equal over two LdgSpans).
     */
    CELER_CEF friend bool operator==(LdgWrapper a, LdgWrapper b) noexcept
    {
        return a.get() == b.get();
    }
    CELER_CEF friend bool operator==(LdgWrapper a, type b) noexcept
    {
        return a.get() == b;
    }
    CELER_CEF friend bool operator==(type a, LdgWrapper b) noexcept
    {
        return a == b.get();
    }
    CELER_CEF friend bool operator!=(LdgWrapper a, LdgWrapper b) noexcept
    {
        return a.get() != b.get();
    }
    CELER_CEF friend bool operator!=(LdgWrapper a, type b) noexcept
    {
        return a.get() != b;
    }
    CELER_CEF friend bool operator!=(type a, LdgWrapper b) noexcept
    {
        return a != b.get();
    }
    //!@}

  private:
    T* ptr_;
};

//---------------------------------------------------------------------------//
/*!
 * Proxy iterator that constructs a LdgWrapper when dereferenced.
 * \tparam T value type being accessed
 *
 * See \c ldg .
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
    using reference = LdgWrapper<T>;
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
        return LdgWrapper<T>{*ptr_};
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
        return LdgWrapper<T>{*(ptr_ + n)};
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
LdgWrapper(T&) -> LdgWrapper<std::add_const_t<T>>;

template<class T>
LdgIterator(T*) -> LdgIterator<std::add_const_t<T>>;

//---------------------------------------------------------------------------//
//! Get the item that's wrapped

template<class T>
CELER_CONSTEXPR_FUNCTION T remove_ldg_wrapper(LdgWrapper<T const> val)
{
    return val.get();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

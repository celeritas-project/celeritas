//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"

#include "detail/LdgIteratorImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Iterator for read-only device data in global memory.
 *
 * This wraps pointer accesses with the \c __ldg intrinsic to load
 * read-only data using texture cache.
 */
template<class T>
class LdgIterator
{
    static_assert(detail::is_ldg_supported_v<T>,
                  "LdgIterator requires const arithmetic, OpaqueId or "
                  "enum type");

  private:
    using LoadPolicyT = detail::LdgLoader<T>;

  public:
    //!@{
    //! \name Type aliases
    using difference_type = std::ptrdiff_t;
    using value_type = typename LoadPolicyT::value_type;
    using pointer = typename LoadPolicyT::pointer;
    using reference = typename LoadPolicyT::reference;
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
        return LoadPolicyT::read(ptr_);
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
        return LoadPolicyT::read(ptr_ + n);
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

  private:
    pointer ptr_{nullptr};
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class T>
LdgIterator(T*) -> LdgIterator<std::add_const_t<T>>;

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
operator==(LdgIterator<T> const& it, std::nullptr_t) noexcept
{
    return !static_cast<bool>(it);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(LdgIterator<T> const& it, std::nullptr_t) noexcept
{
    return static_cast<bool>(it);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator==(std::nullptr_t, LdgIterator<T> const& it) noexcept
{
    return !static_cast<bool>(it);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(std::nullptr_t, LdgIterator<T> const& it) noexcept
{
    return static_cast<bool>(it);
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
          typename LdgIterator<T>::difference_type const n) noexcept
{
    LdgIterator tmp{it};
    return tmp += n;
}
template<class T>
CELER_CONSTEXPR_FUNCTION LdgIterator<T>
operator+(typename LdgIterator<T>::difference_type const n,
          LdgIterator<T> const& it) noexcept
{
    return it + n;
}
template<class T>
CELER_CONSTEXPR_FUNCTION LdgIterator<T>
operator-(LdgIterator<T> const& it,
          typename LdgIterator<T>::difference_type const n) noexcept
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

//---------------------------------------------------------------------------//
/*!
 * Wrapper struct for specializing on types supported by LdgIterator.
 *
 * For example, Span<LdgValue<T>> specialization can internally use
 * LdgIterator. Specializations should refer to LdgValue<T>::value_type to
 * force the template instantiation of LdgValue and type-check T .
 */
template<class T>
struct LdgValue
{
    using value_type = T;
    static_assert(detail::is_ldg_supported_v<T>,
                  "const arithmetic, OpaqueId or enum type "
                  "required");
};

//---------------------------------------------------------------------------//
//! Alias for a Span iterating over values read using __ldg
template<class T, std::size_t Extent = dynamic_extent>
using LdgSpan = Span<LdgValue<T>, Extent>;

//---------------------------------------------------------------------------//
/*!
 * Construct an array from a fixed-size span, removing LdgValue marker.
 *
 * Note: \code make_array(Span<T,N> const&) \endcode is not reused because:
 * 1. Using this overload reads input data using \c __ldg
 * 2. \code return make_array<T, N>(s) \endcode results in segfault (gcc 11.3).
 *    This might be a compiler bug because temporary lifetime should be
 *    extended until the end of the expression and we return a copy...
 */
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION auto make_array(LdgSpan<T, N> const& s)
{
    Array<std::remove_cv_t<T>, N> result{};
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = s[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

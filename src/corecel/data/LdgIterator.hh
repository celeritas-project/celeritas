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
#include "corecel/cont/Span.hh"

#include "LdgTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Whether a type is supported by __ldg
template<class T>
inline constexpr bool is_ldg_supported_v
    = !std::is_void_v<typename LdgTraits<T>::underlying_type>;

//---------------------------------------------------------------------------//
/*!
 * Wrap the low-level CUDA/HIP "load read-only global memory" function.
 *
 * This low-level capability allows improved caching because we're \em
 * promising that the data is not mem. For CUDA the load is cached in
 * L1/texture memory, theoretically improving performance if repeatedly
 * accessed.
 *
 * \warning The target address must be read-only for the lifetime of the
 * kernel. This is generally true for Params data but not State data.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T ldg(T const* ptr)
{
    using TraitsT = LdgTraits<T>;
    using underlying_type = typename TraitsT::underlying_type;
    static_assert(std::is_arithmetic_v<underlying_type>,
                  "Only types with arithmetic underlying types are supported "
                  "by __ldg");

    auto const* data_ptr = TraitsT::data(ptr);
#if CELER_DEVICE_COMPILE
    return T{__ldg(data_ptr)};
#else
    return T{*data_ptr};
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Iterator for read-only device data in global memory.
 * \tparam T value type being accessed
 *
 * This wraps pointer accesses with the \c __ldg intrinsic to load
 * read-only data using texture cache.
 */
template<class T>
class LdgIterator
{
    using TraitsT = LdgTraits<T>;
    static_assert(std::is_const_v<T>,
                  "LDG access can only be performed for constant data");
    static_assert(is_ldg_supported_v<std::remove_const_t<T>>,
                  "LDG access is limited to certain primitive types");

  public:
    //!@{
    //! \name Type aliases
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_const_t<T>;
    using pointer = T*;
    using reference = value_type;
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
        return ldg(ptr_);
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
        return ldg(ptr_ + n);
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
    static_assert(std::is_const_v<T>);
    static_assert(is_ldg_supported_v<std::remove_const_t<T>>,
                  "const arithmetic, OpaqueId or enum type required");
};

//---------------------------------------------------------------------------//
//! Alias for a Span iterating over const values read using __ldg
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

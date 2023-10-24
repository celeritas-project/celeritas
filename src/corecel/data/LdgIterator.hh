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
namespace detail
{
//---------------------------------------------------------------------------//

/*!
 * Reads a value T using __ldg builtin and return a copy of it
 */
template<class T, typename = void>
struct LDGLoad
{
    using value_type = T;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        return __ldg(p);
#else
        return *p;
#endif
    }
};

/*!
 * Specialization when T == OpaqueId.
 * Wraps the underlying index in a OpaqueId when returning it.
 */
template<class T>
struct LDGLoad<T, std::enable_if_t<is_opaqueid_v<T>>>
{
    using value_type = T;
    using pointer = std::add_pointer_t<typename T::size_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        return value_type{__ldg(p)};
#else
        return value_type{*p};
#endif
    }
};
//---------------------------------------------------------------------------//
}  // namespace detail

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
    using LDGLoadPolicy = detail::LDGLoad<T>;

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
    CELER_CONSTEXPR_FUNCTION LdgIterator
    operator-(const difference_type n) const noexcept
    {
        LdgIterator tmp{ptr_};
        return tmp -= n;
    }
    CELER_CONSTEXPR_FUNCTION difference_type
    operator-(LdgIterator const& it) const noexcept
    {
        return it.ptr_ - ptr_;
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
    pointer ptr_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

//!@{
//! RandomAccessIterator requirements
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator==(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs)
{
    using pointer = typename LdgIterator<T>::pointer;
    return static_cast<pointer>(lhs) == static_cast<pointer>(rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs)
{
    return !(lhs == rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator<(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs)
{
    using pointer = typename LdgIterator<T>::pointer;
    return static_cast<pointer>(lhs) < static_cast<pointer>(rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator>(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs)
{
    return rhs < lhs;
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator<=(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs)
{
    return !(lhs > rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator>=(LdgIterator<T> const& lhs, LdgIterator<T> const& rhs)
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
          LdgIterator<T> const& it)
{
    return it + n;
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
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
class LDGIterator
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
    constexpr LDGIterator() noexcept = default;
    constexpr LDGIterator(LDGIterator const&) noexcept = default;
    CELER_CONSTEXPR_FUNCTION LDGIterator(std::nullptr_t) noexcept {}
    CELER_CONSTEXPR_FUNCTION explicit LDGIterator(pointer ptr) noexcept
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
    CELER_CONSTEXPR_FUNCTION LDGIterator& operator++() noexcept
    {
        ++ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION void swap(LDGIterator& it) noexcept
    {
        ::celeritas::trivial_swap(ptr_, it.ptr_);
    }
    CELER_CONSTEXPR_FUNCTION bool operator==(LDGIterator const& it)
    {
        return ptr_ == it.ptr_;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator operator++(int) noexcept
    {
        LDGIterator tmp{ptr_};
        ++ptr_;
        return tmp;
    }
    CELER_CONSTEXPR_FUNCTION pointer operator->() const noexcept
    {
        return ptr_;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator& operator--() noexcept
    {
        --ptr_;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator operator--(int) noexcept
    {
        LDGIterator tmp{ptr_};
        --ptr_;
        return tmp;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator&
    operator+=(const difference_type n) noexcept
    {
        ptr_ += n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator
    operator+(const difference_type n) const noexcept
    {
        LDGIterator tmp{ptr_};
        return tmp += n;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator&
    operator-=(const difference_type n) noexcept
    {
        ptr_ -= n;
        return *this;
    }
    CELER_CONSTEXPR_FUNCTION LDGIterator
    operator-(const difference_type n) const noexcept
    {
        LDGIterator tmp{ptr_};
        return tmp -= n;
    }
    CELER_CONSTEXPR_FUNCTION difference_type
    operator-(LDGIterator const& it) const noexcept
    {
        return it.ptr_ - ptr_;
    }
    CELER_CONSTEXPR_FUNCTION reference
    operator[](const difference_type n) const noexcept
    {
        return LDGLoadPolicy::read(ptr_ + n);
    }
    CELER_CONSTEXPR_FUNCTION friend bool
    operator<(LDGIterator const& lhs, LDGIterator const& rhs)
    {
        return lhs.ptr_ < rhs.ptr_;
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
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(LDGIterator<T> const& lhs, LDGIterator<T> const& rhs)
{
    return !(lhs == rhs);
}
template<class T>
CELER_CONSTEXPR_FUNCTION void
swap(LDGIterator<T>& lhs, LDGIterator<T>& rhs) noexcept
{
    return lhs.swap(rhs);
}
//!@}

//!@{
//! Helper
template<class T>
inline LDGIterator<T> make_ldgiterator(T const* ptr) noexcept
{
    return LDGIterator<T>{ptr};
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ObserverPtr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Type-safe non-owning pointer.
 *
 * This class is based on WG21 N4282, "A Proposal for the World’s Dumbest Smart
 * Pointer, v4". It adds memspace safety similar to Thrust's host/device_ptr.
 *
 * The dereferencing operators can *only* be used from the "native" memspace:
 * i.e., host data can be accessed from a .cc file, and device data from a .cu
 * file.
 *
 * The get() function accesses the pointer with no memspace checking.
 */
template<class T, MemSpace M = MemSpace::native>
class ObserverPtr
{
  public:
    //!@{
    //! \name Type aliases
    using element_type = T;
    using pointer = std::add_pointer_t<T>;
    using reference = std::add_lvalue_reference_t<T>;
    //!@}

  public:
    //!@{
    //! Construct a pointer
    constexpr ObserverPtr() noexcept = default;
    CELER_CONSTEXPR_FUNCTION ObserverPtr(std::nullptr_t) noexcept {}
    CELER_CONSTEXPR_FUNCTION explicit ObserverPtr(pointer ptr) noexcept
        : ptr_{ptr}
    {
    }

    template<class T2>
    CELER_CONSTEXPR_FUNCTION ObserverPtr(ObserverPtr<T2, M> other) noexcept
        : ptr_{other.ptr_}
    {
    }
    //!@}

    //!@{
    //! Access the pointer
    CELER_CONSTEXPR_FUNCTION pointer get() const noexcept { return ptr_; }
    CELER_CONSTEXPR_FUNCTION reference operator*() const noexcept
    {
        return *this->checked_get();
    }
    CELER_CONSTEXPR_FUNCTION pointer operator->() const noexcept
    {
        return this->checked_get();
    }
    CELER_CONSTEXPR_FUNCTION explicit operator pointer() const noexcept
    {
        return this->checked_get();
    }
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const noexcept
    {
        return ptr_ != nullptr;
    }
    //!@}

    //!@{
    //! Modify the pointer
    CELER_CONSTEXPR_FUNCTION pointer release() noexcept
    {
        return ::celeritas::exchange(ptr_, nullptr);
    }
    CELER_CONSTEXPR_FUNCTION void reset(pointer ptr = nullptr) noexcept
    {
        ptr_ = ptr;
    }
    CELER_CONSTEXPR_FUNCTION void swap(ObserverPtr& other) noexcept
    {
        ::celeritas::trivial_swap(ptr_, other.ptr_);
    }
    //!@}

  private:
    pointer ptr_{nullptr};

    CELER_CONSTEXPR_FUNCTION pointer checked_get() const noexcept
    {
        static_assert(M == MemSpace::native, "accessing from invalid memspace");
        return ptr_;
    }

    template<class, MemSpace>
    friend class ObserverPtr;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Swap two pointers
template<class T, MemSpace M>
CELER_CONSTEXPR_FUNCTION void
swap(ObserverPtr<T, M>& lhs, ObserverPtr<T, M>& rhs) noexcept
{
    return lhs.swap(rhs);
}

//---------------------------------------------------------------------------//
//!@{
//! Comparators
#define CELER_DEFINE_OBSPTR_CMP(TOKEN)                                         \
    template<class T1, class T2, MemSpace M>                                   \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                              \
        ObserverPtr<T1, M> const& lhs, ObserverPtr<T2, M> const& rhs) noexcept \
    {                                                                          \
        return lhs.get() TOKEN rhs.get();                                      \
    }
CELER_DEFINE_OBSPTR_CMP(==)
CELER_DEFINE_OBSPTR_CMP(!=)
CELER_DEFINE_OBSPTR_CMP(<)
CELER_DEFINE_OBSPTR_CMP(>)
CELER_DEFINE_OBSPTR_CMP(<=)
CELER_DEFINE_OBSPTR_CMP(>=)
#undef CELER_DEFINE_OBSPTR_CMP

template<class T, MemSpace M>
CELER_CONSTEXPR_FUNCTION bool
operator==(ObserverPtr<T, M> const& lhs, std::nullptr_t) noexcept
{
    return !static_cast<bool>(lhs);
}
template<class T, MemSpace M>
CELER_CONSTEXPR_FUNCTION bool
operator!=(ObserverPtr<T, M> const& lhs, std::nullptr_t) noexcept
{
    return static_cast<bool>(lhs);
}
template<class T, MemSpace M>
CELER_CONSTEXPR_FUNCTION bool
operator==(std::nullptr_t, ObserverPtr<T, M> const& rhs) noexcept
{
    return !static_cast<bool>(rhs);
}
template<class T, MemSpace M>
CELER_CONSTEXPR_FUNCTION bool
operator!=(std::nullptr_t, ObserverPtr<T, M> const& rhs) noexcept
{
    return static_cast<bool>(rhs);
}
//!@}

//---------------------------------------------------------------------------//
//! Create an observer pointer from a pointer in the native memspace.
template<class T>
inline ObserverPtr<T> make_observer(T* ptr) noexcept
{
    return ObserverPtr<T>{ptr};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

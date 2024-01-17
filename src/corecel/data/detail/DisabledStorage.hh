//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/DisabledStorage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Null-op placeholder for "value" container on device.
 */
template<class T>
class DisabledStorage
{
  public:
    //!@{
    //! \name Type aliases
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using SpanT = Span<T>;
    using SpanConstT = Span<T const>;
    //!@}
  public:
    //!@{
    //! Null-op functions that should never be called
    DisabledStorage() { CELER_ASSERT_UNREACHABLE(); }
    explicit DisabledStorage(size_type) { CELER_ASSERT_UNREACHABLE(); }
    CELER_FORCEINLINE_FUNCTION bool empty() const
    {
        CELER_ASSERT_UNREACHABLE();
        return true;
    }
    CELER_FORCEINLINE_FUNCTION size_type size() const
    {
        CELER_ASSERT_UNREACHABLE();
        return 0;
    }
    CELER_FORCEINLINE_FUNCTION pointer data() const
    {
        CELER_ASSERT_UNREACHABLE();
        return nullptr;
    }
    CELER_FORCEINLINE_FUNCTION void copy_to_device(SpanConstT)
    {
        CELER_ASSERT_UNREACHABLE();
    }
    CELER_FORCEINLINE_FUNCTION void copy_to_host(SpanT) const
    {
        CELER_ASSERT_UNREACHABLE();
    }

    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

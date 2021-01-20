//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SpanImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include "base/Assert.hh"
#include "base/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Sentinel value for span of dynamic type
constexpr std::size_t dynamic_extent = std::size_t(-1);

//---------------------------------------------------------------------------//
//! Calculate the return type for a subspan
CELER_CONSTEXPR_FUNCTION std::size_t
subspan_extent(std::size_t extent, std::size_t offset, std::size_t count)
{
    return count != dynamic_extent
               ? count
               : (extent != dynamic_extent ? extent - offset : dynamic_extent);
}

//---------------------------------------------------------------------------//
//! Calculate the size of a subspan
CELER_CONSTEXPR_FUNCTION std::size_t
subspan_size(std::size_t size, std::size_t offset, std::size_t count)
{
    return count != dynamic_extent ? count : size - offset;
}

//---------------------------------------------------------------------------//
/*!
 * Storage for a Span.
 */
template<class T, std::size_t Extent>
struct SpanImpl
{
    //// DATA ////

    T*                           data = nullptr;
    static constexpr std::size_t size = Extent;

    //// METHODS ////

    //! No default constructor for fixed-size type
    SpanImpl() = delete;

    //! Construct from a dynamic span
    CELER_FORCEINLINE_FUNCTION
    SpanImpl(const SpanImpl<T, dynamic_extent>& other) : data(other.data)
    {
        CELER_EXPECT(other.size == Extent);
    }

    //! Construct from data and size
    CELER_FORCEINLINE_FUNCTION SpanImpl(T* d, std::size_t s) : data(d)
    {
        CELER_EXPECT(d != nullptr);
        CELER_EXPECT(s == Extent);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for size-zero span.
 */
template<class T>
struct SpanImpl<T, 0>
{
    //// DATA ////

    T*                           data = nullptr;
    static constexpr std::size_t size = 0;

    //// CONSTRUCTORS ////

    //! Default constructor is empty and size zero
    constexpr SpanImpl() = default;

    //! Construct from data (any) and size (must be zero)
    CELER_FORCEINLINE_FUNCTION SpanImpl(T* d, std::size_t s) : data(d)
    {
        CELER_EXPECT(s == 0);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for dynamic span.
 */
template<class T>
struct SpanImpl<T, dynamic_extent>
{
    //// DATA ////

    T*          data = nullptr;
    std::size_t size = 0;

    //// METHODS ////

    //! Default constructor is empty and size zero
    constexpr SpanImpl() = default;

    //! Construct from data and size
    CELER_FORCEINLINE_FUNCTION SpanImpl(T* d, std::size_t s) : data(d), size(s)
    {
        CELER_EXPECT(d != nullptr || size == 0);
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

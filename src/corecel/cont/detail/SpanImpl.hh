//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/detail/SpanImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
struct LdgValue;
template<class T>
class LdgIterator;

//---------------------------------------------------------------------------//
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Default type aliases for Span.
 */
template<class T>
struct SpanTraits
{
    using element_type = T;
    using pointer = std::add_pointer_t<element_type>;
    using const_pointer = std::add_pointer_t<element_type const>;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reference = std::add_lvalue_reference_t<element_type>;
    using const_reference = std::add_lvalue_reference_t<element_type const>;
};

//---------------------------------------------------------------------------//
/*!
 * Type aliases when data is read using __ldg.
 *
 * \c LdgValue checks that T is a
 * valid type (const arithmetic or OpaqueId). Since \c LdgIterator returns a
 * copy of the data read, we can't return a reference to the original data, we
 * need to return a copy as well.
 */
template<class T>
struct SpanTraits<LdgValue<T>>
{
    using element_type = typename LdgValue<T>::value_type;  // Always const!
    using pointer = std::add_pointer_t<element_type const>;
    using const_pointer = pointer;
    using iterator = LdgIterator<element_type const>;
    using const_iterator = iterator;
    using reference = std::remove_const_t<element_type>;
    using const_reference = std::remove_const_t<element_type>;
};

//---------------------------------------------------------------------------//
//! Sentinel value for span of dynamic type
inline constexpr std::size_t dynamic_extent = std::size_t(-1);

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
    //// TYPES ////

    using iterator = typename SpanTraits<T>::iterator;
    using pointer = typename SpanTraits<T>::pointer;

    //// DATA ////

    iterator data = nullptr;
    static constexpr std::size_t size = Extent;

    //// METHODS ////

    //! No default constructor for fixed-size type
    SpanImpl() = delete;

    //! Construct from a dynamic span
    CELER_FORCEINLINE_FUNCTION
    SpanImpl(SpanImpl<T, dynamic_extent> const& other) : data(other.data)
    {
        CELER_EXPECT(other.size == Extent);
    }

    //! Construct from data and size
    CELER_FORCEINLINE_FUNCTION
    SpanImpl(pointer d, std::size_t s) : data(d)
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
    //// TYPES ////

    using iterator = typename SpanTraits<T>::iterator;
    using pointer = typename SpanTraits<T>::pointer;

    //// DATA ////

    iterator data = nullptr;
    static constexpr std::size_t size = 0;

    //// CONSTRUCTORS ////

    //! Default constructor is empty and size zero
    constexpr SpanImpl() = default;

    //! Construct from data (any) and size (must be zero)
    CELER_FORCEINLINE_FUNCTION
    SpanImpl(pointer d, std::size_t s) : data(d) { CELER_EXPECT(s == 0); }
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for dynamic span.
 */
template<class T>
struct SpanImpl<T, dynamic_extent>
{
    //// TYPES ////

    using iterator = typename SpanTraits<T>::iterator;
    using pointer = typename SpanTraits<T>::pointer;

    //// DATA ////

    iterator data = nullptr;
    std::size_t size = 0;

    //// METHODS ////

    //! Default constructor is empty and size zero
    constexpr SpanImpl() = default;

    //! Construct from data and size
    CELER_FORCEINLINE_FUNCTION
    SpanImpl(pointer d, std::size_t s) : data(d), size(s)
    {
        CELER_EXPECT(d != nullptr || size == 0);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

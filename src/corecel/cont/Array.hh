//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Array.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/StreamableContainer.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fixed-size simple array for storage.
 *
 * The Array class is primarily used for point coordinates (e.g., \c Real3) but
 * is also used for other fixed-size data structures.
 *
 * This is not fully compatible with std::array:
 * - no support for N=0
 * - uses the native celeritas \c size_type (even though this has \em no effect
     on generated code for values of N inside the range of \c size_type
 * - zero-initialized by default
 *
 * \note For supplementary functionality, include:
 * - \c corecel/math/ArrayUtils.hh for real-number vector/matrix applications
 * - \c corecel/math/ArrayOperators.hh for mathematical operators
 * - \c ArrayIO.json.hh for JSON input and output
 */
template<class T, ::celeritas::size_type N>
class Array
{
    static_assert(N > 0);

  public:
    //!@{
    //! \name Type aliases
    using value_type = T;
    using size_type = ::celeritas::size_type;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    using CArrayConstRef = T const (&)[N];
    //!@}

  public:
    //! Default construction initializes to zero
    CELER_CEF Array() : d_{T{}} {}

    //! Construct from an array for aggregate initialization of daughters
    CELER_CEF Array(CArrayConstRef values)
    {
        for (size_type i = 0; i < N; ++i)
        {
            d_[i] = values[i];
        }
    }

    //! Construct with C-style aggregate initialization
    CELER_CEF Array(T first) : d_{first} {}

    //! Construct with the array's data
    template<class... Us>
    CELER_CEF Array(T first, Us... rest) : d_{first, static_cast<T>(rest)...}
    {
        // Protect against leaving off an entry, e.g. Array<int, 2>(1)
        static_assert(sizeof...(rest) + 1 == N,
                      "All array entries must be explicitly specified");
    }

    //// ACCESSORS ////

    //!@{
    //! \name Element access
    CELER_CEF const_reference operator[](size_type i) const { return d_[i]; }
    CELER_CEF reference operator[](size_type i) { return d_[i]; }
    CELER_CEF const_reference front() const { return d_[0]; }
    CELER_CEF reference front() { return d_[0]; }
    CELER_CEF const_reference back() const { return d_[N - 1]; }
    CELER_CEF reference back() { return d_[N - 1]; }
    CELER_CEF const_pointer data() const { return d_; }
    CELER_CEF pointer data() { return d_; }
    //!@}

    //!@{
    //! Access for structured unpacking
    template<std::size_t I>
    CELER_CEF T& get()
    {
        static_assert(I < static_cast<std::size_t>(N));
        return d_[I];
    }
    template<std::size_t I>
    CELER_CEF T const& get() const
    {
        static_assert(I < static_cast<std::size_t>(N));
        return d_[I];
    }
    //!@}

    //!@{
    //! \name Iterators
    CELER_CEF iterator begin() { return d_; }
    CELER_CEF iterator end() { return d_ + N; }
    CELER_CEF const_iterator begin() const { return d_; }
    CELER_CEF const_iterator end() const { return d_ + N; }
    CELER_CEF const_iterator cbegin() const { return d_; }
    CELER_CEF const_iterator cend() const { return d_ + N; }
    //!@}

    //!@{
    //! \name Capacity
    CELER_CEF bool empty() const { return N == 0; }
    static CELER_CEF size_type size() { return N; }
    //!@}

    //!@{
    //! \name  Operations

    //! Fill the array with a constant value
    CELER_CEF void fill(const_reference value)
    {
        for (size_type i = 0; i != N; ++i)
        {
            d_[i] = value;
        }
    }
    //!@}

  private:
    T d_[N];  //!< Storage
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

// Note: this differs from std::array, which deduces from T rather than the
// common type
template<class T, class... Us>
CELER_FUNCTION Array(T, Us...)
    -> Array<std::common_type_t<T, Us...>, 1 + sizeof...(Us)>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Test equality of two arrays.
 */
template<class T, size_type N>
CELER_CEF bool operator==(Array<T, N> const& lhs, Array<T, N> const& rhs)
{
    for (size_type i = 0; i != N; ++i)
    {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Test inequality of two arrays.
 */
template<class T, size_type N>
CELER_CEF bool operator!=(Array<T, N> const& lhs, Array<T, N> const& rhs)
{
    return !(lhs == rhs);
}

#if !CELER_DEVICE_COMPILE
//---------------------------------------------------------------------------//
/*!
 * Write the elements of array \a a to stream \a os.
 */
template<class T, size_type N>
CELER_FORCEINLINE std::ostream&
operator<<(std::ostream& os, Array<T, N> const& a)
{
    os << StreamableContainer{a.data(), a.size()};
    return os;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//---------------------------------------------------------------------------//
//! Support structured binding: array size
template<class T, celeritas::size_type N>
struct tuple_size<celeritas::Array<T, N>>
{
    static constexpr std::size_t value = N;
};

//! Support structured binding: array element type
template<std::size_t I, class T, celeritas::size_type N>
struct tuple_element<I, celeritas::Array<T, N>>
{
    static_assert(I < std::tuple_size<celeritas::Array<T, N>>::value);
    using type = T;
};

//---------------------------------------------------------------------------//
}  // namespace std
//! \endcond

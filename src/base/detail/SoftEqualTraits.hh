//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SoftEqualTraits.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Provide relative errors for soft_equiv based on type.
 *
 * This also gives compile-time checking for bad values.
 */
template<typename T>
struct SoftEqualTraits
{
    using value_type = T;

    //! Default relative error
    static constexpr value_type rel_prec()
    {
        static_assert(sizeof(T) == 0, "Invalid type for softeq!");
        return T();
    }

    //! Default absolute error
    static constexpr value_type abs_thresh()
    {
        static_assert(sizeof(T) == 0, "Invalid type for softeq!");
        return T();
    }
};

template<>
struct SoftEqualTraits<long double>
{
    using value_type = long double;
    static constexpr value_type rel_prec() { return 1.0e-14; }
    static constexpr value_type abs_thresh() { return 1.0e-15; }
};

template<>
struct SoftEqualTraits<double>
{
    using value_type = double;
    static constexpr value_type rel_prec() { return 1.0e-12; }
    static constexpr value_type abs_thresh() { return 1.0e-14; }
};

template<>
struct SoftEqualTraits<float>
{
    using value_type = float;
    static constexpr value_type rel_prec() { return 1.0e-6f; }
    static constexpr value_type abs_thresh() { return 1.0e-8f; }
};

//---------------------------------------------------------------------------//
/*!
 * \struct SoftPrecisionType
 * \brief Get a "least common denominator" for soft comparisons.
 */
template<typename T1, typename T2>
struct SoftPrecisionType
{
    // Equivalent to std::common_type<T1,T2>::type
    using type = decltype(true ? T1() : T2());
};

// When comparing doubles to floats, use the floating point epsilon for
// comparison
template<>
struct SoftPrecisionType<double, float>
{
    using type = float;
};
template<>
struct SoftPrecisionType<float, double>
{
    using type = float;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

//---------------------------------------------------------------------------//

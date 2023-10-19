//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/SoftEqualTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

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
template<class T>
struct SoftEqualTraits
{
    using value_type = T;

    //! Default relative error
    static CELER_CONSTEXPR_FUNCTION value_type rel_prec()
    {
        static_assert(sizeof(T) == 0, "Invalid type for softeq!");
        return T();
    }

    //! Default absolute error
    static CELER_CONSTEXPR_FUNCTION value_type abs_thresh()
    {
        static_assert(sizeof(T) == 0, "Invalid type for softeq!");
        return T();
    }
};

template<>
struct SoftEqualTraits<double>
{
    using value_type = double;
    static CELER_CONSTEXPR_FUNCTION value_type sqrt_prec() { return 1.0e-6; }
    static CELER_CONSTEXPR_FUNCTION value_type rel_prec() { return 1.0e-12; }
    static CELER_CONSTEXPR_FUNCTION value_type abs_thresh() { return 1.0e-14; }
};

template<>
struct SoftEqualTraits<float>
{
    using value_type = float;
    static CELER_CONSTEXPR_FUNCTION value_type sqrt_prec() { return 1.0e-3f; }
    static CELER_CONSTEXPR_FUNCTION value_type rel_prec() { return 1.0e-6f; }
    static CELER_CONSTEXPR_FUNCTION value_type abs_thresh() { return 1.0e-6f; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

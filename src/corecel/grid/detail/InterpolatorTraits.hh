//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/detail/InterpolatorTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/grid/GridTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traits class for interpolating with linear/logarithmic scaling.
 */
template<Interp I, class T>
struct InterpolatorTraits;

template<class T>
struct InterpolatorTraits<Interp::linear, T>
{
    static CELER_CONSTEXPR_FUNCTION T transform(T value) { return value; }
    static CELER_CONSTEXPR_FUNCTION T negate_transformed(T value)
    {
        return -value;
    }
    static CELER_CONSTEXPR_FUNCTION T add_transformed(T left, T right)
    {
        return left + right;
    }
    static CELER_CONSTEXPR_FUNCTION T transform_inv(T value) { return value; }
    static CELER_CONSTEXPR_FUNCTION bool valid_domain(T) { return true; }
};

template<class T>
struct InterpolatorTraits<Interp::log, T>
{
    static CELER_CONSTEXPR_FUNCTION T transform(T value)
    {
        return std::log2(value);
    }
    static CELER_CONSTEXPR_FUNCTION T negate_transformed(T value)
    {
        return T(1) / value;
    }
    static CELER_CONSTEXPR_FUNCTION T add_transformed(T left, T right)
    {
        return transform(left * right);
    }
    static CELER_CONSTEXPR_FUNCTION T transform_inv(T value)
    {
        return std::exp2(value);
    }
    static CELER_CONSTEXPR_FUNCTION bool valid_domain(T value)
    {
        return value > T(0);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

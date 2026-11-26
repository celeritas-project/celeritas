//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/Quantity.hh
//! \sa orange/surf/SphereCentered.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/math/ArrayOperators.hh"  // IWYU pragma: keep
#include "corecel/math/Quantity.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Unit for a linear surface gradient.
 */
struct Linear
{
    static CELER_CONSTEXPR_FUNCTION int value() { return 1; }
    static constexpr char const* label() { return "linear"; }
};

//---------------------------------------------------------------------------//
/*!
 * Unit for a quadratic surface gradient.
 */
struct Quadratic
{
    static CELER_CONSTEXPR_FUNCTION int value() { return 2; }
    static constexpr char const* label() { return "quadratic"; }
};

//---------------------------------------------------------------------------//
/*!
 * Unit for a quartic surface gradient.
 */
struct Quartic
{
    static CELER_CONSTEXPR_FUNCTION int value() { return 4; }
    static constexpr char const* label() { return "quartic"; }
};

//---------------------------------------------------------------------------//
//! A surface gradient scaled by the degree of its defining equation.
template<class UnitT>
using ScaledReal3 = Quantity<UnitT, Real3>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/LengthQuantities.hh
//! \brief Length unit types and quantity aliases for geometry interop
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayQuantity.hh"  // IWYU pragma: export
#include "corecel/math/Quantity.hh"  // IWYU pragma: export

#include "LengthUnits.hh"

namespace celeritas
{
namespace lengthunits
{
//---------------------------------------------------------------------------//
//!@{
//! \name Length unit types

//! Centimeter length (CGS/Gaussian units)
struct Centimeter
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return lengthunits::centimeter;
    }
    static char const* label() { return "cm"; }
};

//! Millimeter length (CLHEP units)
struct Millimeter
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return lengthunits::millimeter;
    }
    static char const* label() { return "mm"; }
};

//!@}

//---------------------------------------------------------------------------//
//!@{
//! \name Length quantity aliases

using CmLength = RealQuantity<Centimeter>;
using ClhepLength = Quantity<Millimeter, double>;

//!@}
//---------------------------------------------------------------------------//
}  // namespace lengthunits
}  // namespace celeritas

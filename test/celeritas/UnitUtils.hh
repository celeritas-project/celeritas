//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/UnitUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Convert a value to centimeters from the native system
constexpr inline real_type to_cm(real_type v)
{
    return native_value_to<units::CmLength>(v).value();
}

//---------------------------------------------------------------------------//
//! Convert a value *from* centimeters to the native system
constexpr inline real_type from_cm(real_type v)
{
    return native_value_from(units::CmLength{v});
}

//---------------------------------------------------------------------------//
//! Convert an array to centimeters from the native system
constexpr inline Array<real_type, 3> to_cm(Array<real_type, 3> const& v)
{
    return {to_cm(v[0]), to_cm(v[1]), to_cm(v[2])};
}

//---------------------------------------------------------------------------//
//! Convert an array *from* centimeters to the native system
constexpr inline Array<real_type, 3> from_cm(Array<real_type, 3> const& v)
{
    return {from_cm(v[0]), from_cm(v[1]), from_cm(v[2])};
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

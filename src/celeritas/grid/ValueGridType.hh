//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridType.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Hardcoded types of grid data
enum class ValueGridType
{
    macro_xs,  //!< Interaction cross sections
    energy_loss,  //!< Energy loss per unit length
    size_
};

template<class T>
using ValueGridArray = EnumArray<ValueGridType, T>;

// Get the string representation of a grid
char const* to_cstring(ValueGridType grid);

//---------------------------------------------------------------------------//
}  // namespace celeritas

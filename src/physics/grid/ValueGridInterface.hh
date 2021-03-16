//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/EnumArray.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Hardcoded types of grid data
enum class ValueGridType
{
    macro_xs,    //!< Interaction cross sections
    energy_loss, //!< Energy loss per unit length
    range,       //!< Particle range
    size_        //!< Sentinel value
};

template<class T>
using ValueGridArray = EnumArray<ValueGridType, T>;

const char* to_cstring(ValueGridType grid);

//---------------------------------------------------------------------------//
} // namespace celeritas

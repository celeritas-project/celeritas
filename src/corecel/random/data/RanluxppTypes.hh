//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/data/RanluxppTypes.hh
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/cont/Array.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
//! 64-bit unsigned integer type for Ranluxpp
using RanluxppUInt = std::uint64_t;

//! Array of unsigned ints of length 9
using RanluxppArray9 = Array<RanluxppUInt, 9>;

//! Array of unsigned ints of length 18
using RanluxppArray18 = Array<RanluxppUInt, 18>;

//---------------------------------------------------------------------------//
}  // namespace celeritas

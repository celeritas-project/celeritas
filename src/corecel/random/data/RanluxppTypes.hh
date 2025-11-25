//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/RanluxppTypes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/cont/Array.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
//! 64-bit unsigned integer type for Ranluxpp.
using RanluxppUInt = std::uint64_t;

//! Array of unsigned ints of length 9.
using RanluxppArray9 = Array<RanluxppUInt, 9>;

//! Array of unsigned ints of length 18.
using RanluxppArray18 = Array<RanluxppUInt, 18>;

//---------------------------------------------------------------------------//
//! Defines a Ranluxpp number (576-bit number plus carry bit)
struct RanluxppNumber
{
    RanluxppArray9 number{};
    unsigned int carry = 0;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Get the "skip" message for the skip macro
const char* skip_cstring();

//---------------------------------------------------------------------------//
// Number of base-10 digits in an unsigned integer
int num_digits(unsigned long val);

//---------------------------------------------------------------------------//
std::string char_to_hex_string(unsigned char value);

//---------------------------------------------------------------------------//
const char*
trunc_string(unsigned int digits, const char* str, const char* trunc);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

//---------------------------------------------------------------------------//

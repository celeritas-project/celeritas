//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Memory.cc
//---------------------------------------------------------------------------//
#include "Memory.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void device_memset(void*, int, size_type)
{
    REQUIRE(0);
}

//---------------------------------------------------------------------------//
} // namespace celeritas

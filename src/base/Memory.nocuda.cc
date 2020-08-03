//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Memory.nocuda.cc
//---------------------------------------------------------------------------//
#include "Memory.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * This function should never be called when CUDA is disabled. It is
 * implemented with an assertion to allow linking as a nullop implementation.
 */
void device_memset(void*, int, size_type)
{
    REQUIRE(0);
}

//---------------------------------------------------------------------------//
} // namespace celeritas

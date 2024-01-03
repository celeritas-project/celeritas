//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/CsgTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/Types.hh"

namespace celeritas
{
class CsgTree;

namespace test
{
//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

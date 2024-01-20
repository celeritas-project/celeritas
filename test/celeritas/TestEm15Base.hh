//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm15Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeantTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for "infinite" geometry.
 */
class TestEm15Base : virtual public GeantTestBase
{
  protected:
    std::string_view geometry_basename() const override { return "testem15"; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

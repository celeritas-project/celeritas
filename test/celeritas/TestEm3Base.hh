//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm3Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeantTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for replicating the AdePT TestEm3 input.
 *
 * This class requires Geant4 to import the data.
 */
class TestEm3Base : public GeantTestBase
{
  protected:
    char const* geometry_basename() const override { return "testem3-flat"; }
    bool enable_fluctuation() const override { return true; }
    bool enable_msc() const override { return false; }
    bool combined_brems() const override { return true; }
    real_type secondary_stack_factor() const override { return 3.0; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

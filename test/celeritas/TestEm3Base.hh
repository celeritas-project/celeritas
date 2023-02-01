//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm3Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/ext/GeantPhysicsOptions.hh"

#include "GeantTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for replicating the AdePT TestEm3 input.
 *
 * This class requires Geant4 to import the data. MSC is on by default.
 */
class TestEm3Base : public GeantTestBase
{
  protected:
    char const* geometry_basename() const override { return "testem3-flat"; }
    bool combined_brems() const override { return true; }
    real_type secondary_stack_factor() const override { return 3.0; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

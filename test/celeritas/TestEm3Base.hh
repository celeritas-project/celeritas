//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm3Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeantTestBase.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for replicating the AdePT TestEm3 input.
 *
 * This class requires Geant4 to import the data.
 *
 * \todo Refactor to allow more generic geant4 problem setup, move similar code
 * with LDemo and Acceleritas into main library.
 */
class TestEm3Base : public GeantTestBase
{
  protected:
    const char* geometry_basename() const override { return "testem3-flat"; }
    bool        enable_fluctuation() const override { return true; }
    bool        enable_msc() const override { return false; }
    real_type   secondary_stack_factor() const override { return 3.0; }
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test

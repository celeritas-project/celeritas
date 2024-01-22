//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm3Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/phys/ProcessBuilder.hh"

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
class TestEm3Base : virtual public GeantTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "testem3-flat";
    }

    ProcessBuilderOptions build_process_options() const override
    {
        ProcessBuilderOptions opts = GeantTestBase::build_process_options();
        opts.brem_combined = true;
        return opts;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

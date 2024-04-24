//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/LArSphereBase.hh
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
 * Test harness for liquid argon sphere with optical properties.
 *
 * This class requires Geant4 to import the data. MSC is on by default.
 */
class LArSphereBase : virtual public GeantTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "lar-sphere";
    }

    ProcessBuilderOptions build_process_options() const override
    {
        ProcessBuilderOptions opts = GeantTestBase::build_process_options();
        return opts;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

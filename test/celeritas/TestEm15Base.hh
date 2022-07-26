//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/TestEm15Base.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeantTestBase.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for "infinite" geometry.
 */
class TestEm15Base : public GeantTestBase
{
  protected:
    const char* geometry_basename() const override { return "testem15"; }
    bool        enable_fluctuation() const override { return true; }
    bool        enable_msc() const override { return true; }
    bool        combined_brems() const override { return false; }
    real_type   secondary_stack_factor() const override { return 3.0; }
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test

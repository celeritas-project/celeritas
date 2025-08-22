//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    std::string_view gdml_basename() const override { return "testem15"; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

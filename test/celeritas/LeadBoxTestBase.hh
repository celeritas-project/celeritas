//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/LeadBoxTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeantTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for large lead box.
 */
class LeadBoxTestBase : virtual public GeantTestBase
{
  protected:
    std::string_view gdml_basename() const override { return "lead-box"; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

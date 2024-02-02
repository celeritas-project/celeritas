//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    std::string_view geometry_basename() const override { return "lead-box"; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

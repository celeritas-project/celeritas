//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SolidsGeoTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "GenericGeoTestInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test the solids geometry.
 */
class SolidsGeoTest
{
  public:
    static std::string_view geometry_basename() { return "solids"; }

    SolidsGeoTest(GenericGeoTestInterface* geo_test);
    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

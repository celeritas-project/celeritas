//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TransformedBoxGeoTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

namespace celeritas
{
namespace test
{
class GenericGeoTestInterface;

//---------------------------------------------------------------------------//
/*!
 * Test the transformed box geometry.
 */
class TransformedBoxGeoTest
{
  public:
    static std::string_view geometry_basename() { return "transformed-box"; }

    TransformedBoxGeoTest(GenericGeoTestInterface* geo_test);
    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

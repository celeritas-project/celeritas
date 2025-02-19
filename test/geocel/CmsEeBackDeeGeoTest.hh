//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CmsEeBackDeeGeoTest.hh
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
 * Test the four-levels geometry.
 */
class CmsEeBackDeeGeoTest
{
  public:
    static std::string_view geometry_basename() { return "cms-ee-back-dee"; }

    CmsEeBackDeeGeoTest(GenericGeoTestInterface* geo_test);
    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

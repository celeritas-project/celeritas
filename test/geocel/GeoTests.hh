//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTests.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/math/Turn.hh"

#include "GenericGeoTestInterface.hh"
#include "TestMacros.hh"

// TODO: move the classes below into this file
#include "CmsEeBackDeeGeoTest.hh"
#include "CmseGeoTest.hh"
#include "FourLevelsGeoTest.hh"
#include "MultiLevelGeoTest.hh"
#include "SolidsGeoTest.hh"
#include "TransformedBoxGeoTest.hh"
#include "ZnenvGeoTest.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test the B5 (replica) geometry.
 */
class ReplicaGeoTest
{
  public:
    static std::string_view geometry_basename() { return "replica"; }

    //! Construct with a reference to the GoogleTest
    ReplicaGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;
    void test_volume_stack() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

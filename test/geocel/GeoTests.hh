//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTests.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "geocel/Types.hh"

#include "TestMacros.hh"
#include "UnitUtils.hh"

namespace celeritas
{
namespace test
{
class GenericGeoTestInterface;

//---------------------------------------------------------------------------//
/*!
 * Test the CMS EE (reflecting) geometry.
 */
class CmsEeBackDeeGeoTest
{
  public:
    static std::string_view geometry_basename() { return "cms-ee-back-dee"; }

    //! Construct with a reference to the GoogleTest
    CmsEeBackDeeGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the CMS polycone geometry.
 */
class CmseGeoTest
{
  public:
    static std::string_view geometry_basename() { return "cmse"; }

    //! Construct with a reference to the GoogleTest
    CmseGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the four-levels geometry.
 */
class FourLevelsGeoTest
{
  public:
    static std::string_view geometry_basename() { return "four-levels"; }

    //! Construct with a reference to the GoogleTest
    FourLevelsGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the multi-level geometry.
 */
class MultiLevelGeoTest
{
  public:
    static std::string_view geometry_basename() { return "multi-level"; }

    //! Construct with a reference to the GoogleTest
    MultiLevelGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

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
/*!
 * Test the solids geometry.
 */
class SolidsGeoTest
{
  public:
    static std::string_view geometry_basename() { return "solids"; }

    //! Construct with a reference to the GoogleTest
    SolidsGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the transformed box geometry.
 */
class TransformedBoxGeoTest
{
  public:
    static std::string_view geometry_basename() { return "transformed-box"; }

    //! Construct with a reference to the GoogleTest
    TransformedBoxGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test}
    {
    }

    void test_accessors() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the two-box geometry.
 */
class TwoBoxesGeoTest
{
  public:
    static std::string_view geometry_basename() { return "two-boxes"; }

    //! Construct with a reference to the GoogleTest
    TwoBoxesGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_trace() const;

    template<class GeoTest>
    inline static void test_detailed_tracking(GeoTest* geo_test);

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the transformed box geometry.
 */
class ZnenvGeoTest
{
  public:
    static std::string_view geometry_basename() { return "znenv"; }

    //! Construct with a reference to the GoogleTest
    ZnenvGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
// INLINE TEMPLATE TESTS
//---------------------------------------------------------------------------//

template<class GeoTest>
void TwoBoxesGeoTest::test_detailed_tracking(GeoTest* test)
{
    auto geo = test->make_geo_track_view({0, 0, 0}, {0, 0, 1});
    EXPECT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", test->volume_name(geo));

    // Shouldn't hit boundary
    auto next = geo.find_next_step(from_cm(1.25));
    EXPECT_SOFT_EQ(1.25, to_cm(next.distance));
    EXPECT_FALSE(next.boundary);

    geo.move_internal(from_cm(1.25));
    real_type expected_safety = 5 - 1.25;
    EXPECT_SOFT_NEAR(
        expected_safety, to_cm(geo.find_safety()), test->safety_tol());

    // Change direction and try again (hit)
    geo.set_dir({1, 0, 0});
    next = geo.find_next_step(from_cm(50));
    EXPECT_SOFT_EQ(5, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_FALSE(geo.is_outside());
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("world", test->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({5, 0, 1.25}), to_cm(geo.pos()));
    if (geo.is_on_boundary() && CELERITAS_DEBUG)
    {
        // Don't check the safety distance on the boundary; we know by
        // definition it's zero
        EXPECT_THROW(geo.find_safety(), DebugError);
    }

    // Scatter to tangent along boundary
    geo.set_dir({1e-8, 1, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(500, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_internal(from_cm(2));

    // Scatter back inside
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_TRUE(next.boundary);
    EXPECT_SOFT_NEAR(2e-8, to_cm(next.distance), 1e-4);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", test->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({5, 2, 1.25}), to_cm(geo.pos()));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

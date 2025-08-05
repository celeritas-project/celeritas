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

    void test_model() const;
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

    void test_model() const;
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

    void test_model() const;
    void test_accessors() const;
    void test_trace() const;

    template<class GeoTest>
    inline static void test_detailed_tracking(GeoTest* geo_test);

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

    void test_model() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the optical surfaces geometry.
 */
class OpticalSurfacesGeoTest
{
  public:
    static std::string_view geometry_basename() { return "optical-surfaces"; }

    //! Construct with a reference to the GoogleTest
    OpticalSurfacesGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test}
    {
    }

    void test_model() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test a bunch of polyhedra.
 */
class PolyhedraGeoTest
{
  public:
    static std::string_view geometry_basename() { return "polyhedra"; }

    //! Construct with a reference to the GoogleTest
    PolyhedraGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_model() const;
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

    void test_model() const;
    void test_trace() const;
    void test_volume_stack() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the simple CMS geometry.
 */
class SimpleCmsGeoTest
{
  public:
    static std::string_view geometry_basename() { return "simple-cms"; }

    //! Construct with a reference to the GoogleTest
    SimpleCmsGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_model() const;
    void test_trace() const;

    template<class GeoTest>
    inline static void test_detailed_tracking(GeoTest* geo_test);

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
    void test_model() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test TestEm3 (nested structures).
 */
class TestEm3GeoTest
{
  public:
    static std::string_view geometry_basename() { return "testem3"; }

    //! Construct with a reference to the GoogleTest
    TestEm3GeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_model() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test TestEm3 (flattened).
 */
class TestEm3FlatGeoTest
{
  public:
    static std::string_view geometry_basename() { return "testem3-flat"; }

    //! Construct with a reference to the GoogleTest
    TestEm3FlatGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test tilecal plug.
 */
class TilecalPlugGeoTest
{
  public:
    static std::string_view geometry_basename() { return "tilecal-plug"; }

    //! Construct with a reference to the GoogleTest
    TilecalPlugGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_model() const;
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
    void test_model() const;
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
    void test_model() const;
    void test_trace() const;

    template<class GeoTest>
    inline static void test_detailed_tracking(GeoTest* geo_test);

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the ALICE ZDC (parameterised) geometry.
 */
class ZnenvGeoTest
{
  public:
    static std::string_view geometry_basename() { return "znenv"; }

    //! Construct with a reference to the GoogleTest
    ZnenvGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_model() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
// INLINE TEMPLATE TESTS
//---------------------------------------------------------------------------//

template<class GeoTest>
void FourLevelsGeoTest::test_detailed_tracking(GeoTest* test)
{
    {
        SCOPED_TRACE("rightward along corner");
        auto geo = test->make_geo_track_view({-10, -10, -10}, {1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape2", test->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        auto next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        geo.move_internal(from_cm(3.5));
        EXPECT_FALSE(geo.is_on_boundary());

        // Find one a bit further, then cross it
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(1.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_EQ("Shape2", test->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("Shape1", test->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Find the next boundary and make sure that nearer distances aren't
        // accepted
        next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        next = geo.find_next_step(from_cm(0.5));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("outside in");
        auto geo = test->make_geo_track_view({-25, 6.5, 6.5}, {1, 0, 0});
        EXPECT_TRUE(geo.is_outside());

        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ("World", test->volume_name(geo));
    }
    {
        SCOPED_TRACE("inside out");
        auto geo = test->make_geo_track_view({-23.5, 6.5, 6.5}, {-1, 0, 0});
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ("World", test->volume_name(geo));

        auto next = geo.find_next_step(from_cm(2));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_TRUE(geo.is_outside());

        next = geo.find_next_step();
        EXPECT_GT(next.distance, 1e10);
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("reentrant boundary");

        bool const is_surface_vg = (CELERITAS_VECGEOM_SURFACE
                                    && test->geometry_type() == "VecGeom");

        auto geo = test->make_geo_track_view({15.5, 10, 10}, {-1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape1", test->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces: we should hit the outside of the sphere Shape2
        auto next = geo.find_next_step(from_cm(1.0));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        // Move to the boundary but scatter perpendicularly, away from the
        // sphere
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({0, 1, 0});
        EXPECT_TRUE(geo.is_on_boundary());
        EXPECT_EQ("Shape1", test->volume_name(geo));

        // Move a bit internally, then scatter back toward the sphere
        next = geo.find_next_step(from_cm(10.0));
        EXPECT_SOFT_EQ(6, to_cm(next.distance));
        geo.set_dir({-1, 0, 0});
        EXPECT_EQ("Shape1", test->volume_name(geo));

        // Move to the sphere boundary then scatter still into the sphere
        next = geo.find_next_step(from_cm(10.0));
        // TODO: investigate near-zero movement
        auto expected_distance = to_cm(is_surface_vg ? 1e-13 : 1e-8);
        EXPECT_SOFT_EQ(expected_distance, next.distance);
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({0, -1, 0});
        EXPECT_TRUE(geo.is_on_boundary());
        geo.cross_boundary();
        EXPECT_EQ("Shape2", test->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Travel nearly tangent to the right edge of the sphere, then scatter
        // to still outside
        // TODO: understand difference in distance for surface implementation
        next = geo.find_next_step(from_cm(1.0));
        EXPECT_SOFT_EQ(is_surface_vg ? 9.9737647358664937e-07
                                     : 0.00031622777925735285,
                       to_cm(next.distance));
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({1, 0, 0});
        EXPECT_TRUE(geo.is_on_boundary());
        geo.cross_boundary();
        EXPECT_EQ("Shape1", test->volume_name(geo));

        EXPECT_TRUE(geo.is_on_boundary());
        next = geo.find_next_step(from_cm(10.0));
    }
}

//---------------------------------------------------------------------------//

template<class GeoTest>
void SimpleCmsGeoTest::test_detailed_tracking(GeoTest* test)
{
    auto safety_tol = test->safety_tol();
    auto geo = test->make_geo_track_view({0, 0, 0}, {0, 0, 1});
    EXPECT_EQ("vacuum_tube", test->volume_name(geo));

    auto next = geo.find_next_step(from_cm(100));
    EXPECT_SOFT_EQ(100, to_cm(next.distance));
    EXPECT_FALSE(next.boundary);
    geo.move_internal(from_cm(20));
    EXPECT_SOFT_NEAR(30, to_cm(geo.find_safety()), safety_tol);

    geo.set_dir({1, 0, 0});
    next = geo.find_next_step(from_cm(50));
    EXPECT_SOFT_EQ(30, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    geo.move_to_boundary();
    EXPECT_FALSE(geo.is_outside());
    geo.cross_boundary();
    EXPECT_EQ("si_tracker", test->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({30, 0, 20}), to_cm(geo.pos()));

    // Scatter to tangent
    geo.set_dir({0, 1, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(121.34661099511597, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_internal(from_cm(10));
    EXPECT_SOFT_NEAR(1.6227766016837926, to_cm(geo.find_safety()), safety_tol);

    // Move to boundary and scatter back inside
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(111.34661099511597, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(60., to_cm(next.distance));
}

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
    constexpr real_type dx
        = (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE ? 1e-8 : 1e-4);
    geo.set_dir({dx, 1, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(500, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_internal(from_cm(2));

    // Scatter back inside
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_TRUE(next.boundary);
    EXPECT_SOFT_NEAR(2 * dx, to_cm(next.distance), 1e-4);
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

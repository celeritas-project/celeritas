//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTests.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "geocel/Types.hh"

#include "GenericGeoResults.hh"
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
    static std::string_view gdml_basename() { return "cms-ee-back-dee"; }

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
    static std::string_view gdml_basename() { return "cmse"; }

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
    static std::string_view gdml_basename() { return "four-levels"; }

    //! Construct with a reference to the GoogleTest
    FourLevelsGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_consecutive_compute() const;
    void test_detailed_tracking() const;
    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
/*!
 * Test the LAr sphere.
 */
class LarSphereGeoTest
{
  public:
    static std::string_view gdml_basename() { return "lar-sphere"; }

    //! Construct with a reference to the GoogleTest
    LarSphereGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;
    void test_volume_stack() const;

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
    static std::string_view gdml_basename() { return "multi-level"; }

    //! Construct with a reference to the GoogleTest
    MultiLevelGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;
    void test_volume_stack() const;

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
    static std::string_view gdml_basename() { return "optical-surfaces"; }

    //! Construct with a reference to the GoogleTest
    OpticalSurfacesGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test}
    {
    }

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
    static std::string_view gdml_basename() { return "polyhedra"; }

    //! Construct with a reference to the GoogleTest
    PolyhedraGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

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
    static std::string_view gdml_basename() { return "replica"; }

    //! Construct with a reference to the GoogleTest
    ReplicaGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

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
    static std::string_view gdml_basename() { return "simple-cms"; }

    //! Construct with a reference to the GoogleTest
    SimpleCmsGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;
    void test_detailed_tracking() const;

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
    static std::string_view gdml_basename() { return "solids"; }

    //! Construct with a reference to the GoogleTest
    SolidsGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
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
    static std::string_view gdml_basename() { return "testem3"; }

    //! Construct with a reference to the GoogleTest
    TestEm3GeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

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
    static std::string_view gdml_basename() { return "testem3-flat"; }

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
    static std::string_view gdml_basename() { return "tilecal-plug"; }

    //! Construct with a reference to the GoogleTest
    TilecalPlugGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

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
    static std::string_view gdml_basename() { return "transformed-box"; }

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
    static std::string_view gdml_basename() { return "two-boxes"; }

    //! Construct with a reference to the GoogleTest
    TwoBoxesGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_accessors() const;
    void test_trace() const;
    void test_detailed_tracking() const;

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
    static std::string_view gdml_basename() { return "znenv"; }

    //! Construct with a reference to the GoogleTest
    ZnenvGeoTest(GenericGeoTestInterface* geo_test) : test_{geo_test} {}

    void test_trace() const;

  private:
    GenericGeoTestInterface* test_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

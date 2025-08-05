//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoParameterizedTest.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "GenericGeoTestInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Instantiate a test harness using one of the GeoTest classes.
 *
 * This allows a geometry-specific test (e.g., GeantGeoTest) to be crossed with
 * test results for a specific geometry (e.g., the four-levels results defined
 * in FourLevelsGeoTest).
 *
 * Example:
 * \code
using MultiLevelTest = GenericGeoParameterizedTest<GeantGeoTest,
MultiLevelGeoTest>;

TEST_F(MultiLevelTest, accessors)
{
    this->impl().test_accessors();
}
\endcode
 *
 */
template<class TestBase, class GeoTestImpl>
class GenericGeoParameterizedTest : public TestBase
{
    static_assert(std::is_base_of_v<GenericGeoTestInterface, TestBase>);

  protected:
    using TestImpl = GeoTestImpl;

    std::string_view gdml_basename() const final
    {
        return TestImpl::gdml_basename();
    }

    TestImpl impl() { return TestImpl(this); }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

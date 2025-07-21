//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/InvalidOrangeTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "SimpleTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Construct a test name that is disabled unless ORANGE is the core geometry
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    define TEST_IF_CELERITAS_ORANGE(name) name
#else
#    define TEST_IF_CELERITAS_ORANGE(name) DISABLED_##name
#endif

//---------------------------------------------------------------------------//
/*!
 * Create an ORANGE geometry with errors.
 *
 * Use the \c TEST_IF_CELERITAS_ORANGE when subclassing this test.
 *
 * This is a simple geometry with five spheres, all 1cm radius except for the
 * world which is 10 and outer shell which is 15:
 * - Interior at {0, 0, 0} with "aluminum"
 * - World shell at {0, 0, 0} with vacuum
 * - Missing (incorrectly constructed/buggy) region at {-5, 0, 0}
 * - Different region, also aluminum at {0, 0, 0}
 * - Valid region but missing material at {5, 0, 0}
 *
 * The cross sections are fictional and only gammas are defined: see \c
 * SimpleTestBase .
 */
class InvalidOrangeTestBase : virtual public SimpleTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "<in-memory>";
    }
    SPConstCoreGeo build_geometry() override;
    SPConstGeoMaterial build_geomaterial() override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

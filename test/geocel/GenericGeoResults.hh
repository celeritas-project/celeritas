//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoResults.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/cont/Span.hh"
#include "geocel/Types.hh"

// DEPRECATED: remove in v0.7
#define EXPECT_RESULT_EQ(EXPECTED, ACTUAL) EXPECT_REF_EQ(EXPECTED, ACTUAL)
#define EXPECT_RESULT_NEAR(EXPECTED, ACTUAL, TOL) \
    EXPECT_REF_NEAR(EXPECTED, ACTUAL, TOL)

namespace celeritas
{
class GeoParamsInterface;

namespace test
{
class GenericGeoTestInterface;

//---------------------------------------------------------------------------//
// TRACKING RESULT
//---------------------------------------------------------------------------//

//! Get detailed results from tracking from one cell to the next
struct GenericGeoTrackingResult
{
    std::vector<std::string> volumes;
    std::vector<std::string> volume_instances;
    std::vector<real_type> distances;  //!< [cm]
    std::vector<real_type> halfway_safeties;  //!< [cm]
    // Locations the particle had a very tiny distance in a volume
    std::vector<real_type> bumps;  //!< [cm * 3]

    void print_expected() const;
};

//! Loosen strictness for tracking comparison
struct GenericGeoTrackingTolerance
{
    real_type distance{0};
    real_type safety{0};

    static GenericGeoTrackingTolerance
    from_test(GenericGeoTestInterface const&);
};

// Compare tracking results
::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   char const* tol_expr,
                                   GenericGeoTrackingResult const& expected,
                                   GenericGeoTrackingResult const& actual,
                                   GenericGeoTrackingTolerance const& tol);

//! Compare tracking results with default tolerance
inline ::testing::AssertionResult
IsRefEq(char const* expected_expr,
        char const* actual_expr,
        GenericGeoTrackingResult const& expected,
        GenericGeoTrackingResult const& actual)
{
    return IsRefEq(expected_expr, actual_expr, "default", expected, actual, {});
}

//---------------------------------------------------------------------------//
// STACK RESULT
//---------------------------------------------------------------------------//

//! Get the volume instances and replica IDs from a point
struct GenericGeoVolumeStackResult
{
    std::vector<std::string> volume_instances;
    std::vector<int> replicas;

    static GenericGeoVolumeStackResult
    from_span(GeoParamsInterface const&, Span<VolumeInstanceId const>);
    void print_expected() const;
};

::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   GenericGeoVolumeStackResult const& expected,
                                   GenericGeoVolumeStackResult const& actual);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

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

//! Check that a surface normal is equivalent (modulo the sign)
#define EXPECT_NORMAL_EQUIV(expected, actual) \
    EXPECT_PRED_FORMAT2(::celeritas::test::IsNormalEquiv, expected, actual)

namespace celeritas
{
template<class T>
class LabelIdMultiMap;
namespace inp
{
struct Model;
}

namespace test
{
class GenericGeoTestInterface;

//---------------------------------------------------------------------------//
// Test whether two surface normals are about the same, modulo sign
::testing::AssertionResult IsNormalEquiv(char const* expected_expr,
                                         char const* actual_expr,
                                         Real3 const& expected,
                                         Real3 const& actual);

//---------------------------------------------------------------------------//
// TRACKING RESULT
//---------------------------------------------------------------------------//

//! Get detailed results from tracking from one cell to the next
struct GenericGeoTrackingResult
{
    std::vector<std::string> volumes;
    std::vector<std::string> volume_instances;
    std::vector<real_type> distances;  //!< [cm]
    std::vector<real_type> dot_normal;  //!< [cos theta]
    std::vector<real_type> halfway_safeties;  //!< [cm]
    // Locations the particle had a very tiny distance in a volume
    std::vector<real_type> bumps;  //!< [cm * 3]

    //// STATIC HELPER FUNCTIONS ////

    //! Sentinel value for dot_normal when not on surface
    static constexpr real_type no_surface_normal
        = std::numeric_limits<real_type>::infinity();

    //// HELPER FUNCTIONS ////

    // Delete dot_normals that are all 1
    void clear_boring_normals();

    // Replace dot-normals with a sentinel value
    void disable_surface_normal();

    // Whether surface normals are disabled
    bool disabled_surface_normal() const;

    //! Add a failure sentinel at the end
    void fail() { this->fail_at(volumes.size()); }

    // Add a failure sentinel at a certain index
    void fail_at(std::size_t index);

    // Print expected expression to cout
    void print_expected() const;
};

//! Loosen strictness for tracking comparison
struct GenericGeoTrackingTolerance
{
    real_type distance{0};
    real_type normal{0};
    real_type safety{0};
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
    using LabelMap = LabelIdMultiMap<VolumeInstanceId>;

    std::vector<std::string> volume_instances;

    static GenericGeoVolumeStackResult
    from_span(LabelMap const&, Span<VolumeInstanceId const>);
    void print_expected() const;

    //! Add a failure sentinel at the end
    void fail();
};

::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   GenericGeoVolumeStackResult const& expected,
                                   GenericGeoVolumeStackResult const& actual);

//---------------------------------------------------------------------------//
// MODEL INPUT RESULT
//---------------------------------------------------------------------------//

//! Get the unfolded geometry model input
struct GenericGeoModelInp
{
    struct
    {
        std::vector<std::string> labels;
        std::vector<int> materials;
        std::vector<std::vector<int>> daughters;
    } volume;
    struct
    {
        std::vector<std::string> labels;
        std::vector<int> volumes;
    } volume_instance;
    std::string world;

    struct
    {
        std::vector<std::string> labels;
        std::vector<std::string> volumes;
    } surface;

    struct
    {
        std::vector<std::string> labels;
        std::vector<std::vector<int>> volumes;
    } region;

    struct
    {
        std::vector<std::string> labels;
        std::vector<std::vector<int>> volumes;
    } detector;

    static GenericGeoModelInp from_model_input(inp::Model const& in);
    void print_expected() const;
};

//---------------------------------------------------------------------------//

::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   GenericGeoModelInp const& expected,
                                   GenericGeoModelInp const& actual);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

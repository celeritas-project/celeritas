//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectTestResult.hh
//---------------------------------------------------------------------------//
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "geocel/BoundingBox.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace test
{
class AssertionHelper;
}

namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * This helper struct is returned by IntersectRegionTest's harness.
 *
 * It embeds all the meaningful output from constructing an intersect region:
 * the logical definition, surfaces, and bounding boxes.
 *
 * Use the imported \c print_expected function to generate a copy-pastable
 * \c IntersectTestResult that can be compared using \c EXPECT_REF_EQ.
 */
struct IntersectTestResult
{
    static constexpr char class_str[] = "IntersectTestResult";

    std::string node;
    std::vector<std::string> surfaces;
    BBox interior;
    BBox exterior;

    // Note: resulting node is for additional test harness diagnostics, not ref
    // comparison
    NodeId node_id;

    friend std::ostream& operator<<(std::ostream&, IntersectTestResult const&);

    // Compare with EXPECT_REF_EQ
    friend bool is_ref_eq(::celeritas::test::AssertionHelper& helper,
                          IntersectTestResult const& expected,
                          IntersectTestResult const& actual);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectTestResult.hh
//---------------------------------------------------------------------------//
#pragma once
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "geocel/BoundingBox.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
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
 * Use \c print_expected to generate output to copy-paste into a test; this
 * output defines a \c IntersectTestResult that can be compared using \c
 * EXPECT_REF_EQ.
 */
struct IntersectTestResult
{
    std::string node;
    std::vector<std::string> surfaces;
    BBox interior;
    BBox exterior;

    // Note: resulting node is for additional test harness diagnostics, not ref
    // comparison
    NodeId node_id;

    void print_expected() const;
};

// Compare with EXPECT_REF_EQ
::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   IntersectTestResult const& val1,
                                   IntersectTestResult const& val2);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

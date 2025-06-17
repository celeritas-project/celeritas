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

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   IntersectTestResult const& val1,
                                   IntersectTestResult const& val2);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AssertionHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Store and update an assertion result.
 *
 * This is used for IsRefEq implementations. For example:
 * \code
    AssertionHelper result(expr1, expr2);

    if (a.foo != b.foo)
    {
        result.fail() "  foo: " << a.foo << " != " << b.foo;
    }
    return result;
 *  \endcode
 */
class AssertionHelper
{
  public:
  // Construct with expected/actual expressions
    AssertionHelper(char const* expected_expr, char const* actual_expr);

    // Fail and return a streamable object
    ::testing::AssertionResult& fail();

    // Check the sizes, returning success, adding failure message if not
    bool equal_size(std::size_t expected, std::size_t actual);

    //! Get the streamable assertion result
    operator ::testing::AssertionResult const&() const { return result_; }
    operator ::testing::AssertionResult&& () && { return std::move(result_); }

  private:
    ::testing::AssertionResult result_;
    char const* expected_expr_;
    char const* actual_expr_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

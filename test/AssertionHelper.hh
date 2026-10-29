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
        result.fail() << "  foo: " << a.foo << " != " << b.foo;
    }
    is_attr_eq(result, ...);
    is_soft_eq(result, ...);
    is_vec_soft_eq(result, "bar", a.bar, b.bar);
    return result;
 *  \endcode
 *
 * \todo This class is currently shared by standalone and nested class types,
 * and it really needs to be refactored into a local result plus a nested stack
 * of context.
 */
class AssertionHelper
{
  public:
    // Construct with expected/actual expressions
    AssertionHelper(char const* expected_expr,
                    char const* actual_expr,
                    char const* tol_expr = nullptr);

    // Fail and return a streamable object
    ::testing::AssertionResult& fail();

    // Check the sizes, returning success, adding failure message if not
    bool equal_size(std::size_t expected, std::size_t actual);

    //! Get the streamable assertion result
    operator ::testing::AssertionResult const&() const { return result_; }
    operator ::testing::AssertionResult&&() && { return std::move(result_); }

    //! Get the expected expression string
    char const* expected_expr() const { return expected_expr_; }

    //! Get the actual expression string
    char const* actual_expr() const { return actual_expr_; }

    //! Get the tolerance expression string, if provided
    char const* tol_expr() const { return tol_expr_; }

  private:
    ::testing::AssertionResult result_;
    char const* expected_expr_;
    char const* actual_expr_;
    char const* tol_expr_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

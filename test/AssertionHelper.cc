//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AssertionHelper.cc
//---------------------------------------------------------------------------//
#include "AssertionHelper.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with expected/actual expressions.
 */
AssertionHelper::AssertionHelper(char const* expected_expr,
                                 char const* actual_expr)
    : result_(::testing::AssertionSuccess())
    , expected_expr_(expected_expr)
    , actual_expr_(actual_expr)
{
}

//---------------------------------------------------------------------------//
/*!
 * Fail and return a streamable object.
 */
::testing::AssertionResult& AssertionHelper::fail()
{
    if (result_)
    {
        result_ = ::testing::AssertionFailure();
        result_ << "Expected: (" << expected_expr_ << ") == (" << actual_expr_
                << "), but\n";
    }
    else
    {
        result_ << '\n';
    }
    return result_;
}

//---------------------------------------------------------------------------//
/*!
 * Check the sizes, returning success, adding failure message if not.
 */
bool AssertionHelper::equal_size(std::size_t expected, std::size_t actual)
{
    if (expected != actual)
    {
        this->fail() << " size differs: " << expected << " != " << actual;
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

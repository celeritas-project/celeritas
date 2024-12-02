//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ValidationUtils.cc
//---------------------------------------------------------------------------//
#include "ValidationUtils.hh"

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Template specialization to compare \c ImportPhysicsVector for equivalence.
 *
 * Uses exact comparison to compare imported vectors that have been copied
 * or referenced, and therefore should have the same bit representation as
 * the source vectors.
 */
template<>
::testing::AssertionResult IsVecEq<ImportPhysicsVector, ImportPhysicsVector>(
    char const* expected_expr,
    char const* actual_expr,
    ImportPhysicsVector const& expected,
    ImportPhysicsVector const& actual)
{
    auto x_result = IsVecEq(expected_expr, actual_expr, expected.x, actual.x);
    auto y_result = IsVecEq(expected_expr, actual_expr, expected.y, actual.y);

    ::testing::AssertionResult result(x_result && y_result);
    if (!x_result)
    {
        result << "x values:\n" << x_result.message();
    }
    if (!y_result)
    {
        result << "y values:\n" << y_result.message();
    }

    return result;
}

template<>
::testing::AssertionResult
IsVecEq<std::vector<ImportPhysicsVector>,
        std::vector<std::tuple<Span<real_type const>, Span<real_type const>>>>(
    char const* expected_expr,
    char const* actual_expr,
    std::vector<ImportPhysicsVector> const& expected,
    std::vector<std::tuple<Span<real_type const>, Span<real_type const>>> const&
        actual)
{
    if (expected.size() != actual.size())
    {
        ::testing::AssertionResult failure = ::testing::AssertionFailure();

        failure << " Size of: " << actual_expr
                << "\n  Actual: " << actual.size()
                << "\nExpected: " << expected_expr
                << ".size()\nWhich is: " << expected.size() << "\n";
        return failure;
    }

    ::testing::AssertionResult result = ::testing::AssertionSuccess();

    for (auto i : range(expected.size()))
    {
        auto const& expected_vec = expected[i];
        auto const& [actual_x, actual_y] = actual[i];

        std::string index_expr = "[" + std::to_string(i) + "]";
        std::string expected_expr_i = expected_expr + index_expr;
        std::string actual_expr_i = actual_expr + index_expr;

        auto x_result = IsVecEq(expected_expr_i.c_str(),
                                actual_expr_i.c_str(),
                                expected_vec.x,
                                actual_x);
        auto y_result = IsVecEq(expected_expr_i.c_str(),
                                actual_expr_i.c_str(),
                                expected_vec.y,
                                actual_y);

        if (result && (!x_result || !y_result))
        {
            result = ::testing::AssertionFailure();
        }

        if (!x_result)
        {
            result << "x values:\n" << x_result.message();
        }
        if (!y_result)
        {
            result << "y values:\n" << y_result.message();
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail

namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Construct validator for with the underlying storage.
 */
GridValidator::GridValidator(Items<real_type>* reals, Items<Grid>* grids)
    : reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals_);
    CELER_EXPECT(grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an MFP builder with the underlying collections.
 */
MfpBuilder GridValidator::create_mfp_builder()
{
    return MfpBuilder(reals_, grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with internal collections.
 */
OwningGridValidator::OwningGridValidator() : GridValidator(&reals_, &grids_) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/IsRefEq.cc
//---------------------------------------------------------------------------//
#include "IsRefEq.hh"

#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/BoundingBox.hh"

#include "AssertionHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T>
::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   char const* tol_expr,
                                   BoundingBox<T> const& expected,
                                   BoundingBox<T> const& actual,
                                   T tol)
{
    test::AssertionHelper result{expected_expr, actual_expr};

    if (!expected || !actual)
    {
        if (static_cast<bool>(expected) != static_cast<bool>(actual))
        {
            result.fail() << "Expected " << repr(expected) << " but got "
                          << repr(actual) << " (tolerance " << tol_expr << "="
                          << tol << ')';
        }
        return result;
    }

    auto const points_are_close
        = [tol](auto const& expected_point, auto const& actual_point) {
              return expected_point == actual_point
                     || distance(expected_point, actual_point) <= tol;
          };
    if (!points_are_close(expected.lower(), actual.lower())
        || !points_are_close(expected.upper(), actual.upper()))
    {
        result.fail() << "Expected " << repr(expected) << " but got "
                      << repr(actual) << " (tolerance " << tol_expr << "="
                      << tol << ')';
    }
    return result;
}

template ::testing::AssertionResult IsRefEq<float>(char const*,
                                                   char const*,
                                                   char const*,
                                                   BoundingBox<float> const&,
                                                   BoundingBox<float> const&,
                                                   float);
template ::testing::AssertionResult IsRefEq<double>(char const*,
                                                    char const*,
                                                    char const*,
                                                    BoundingBox<double> const&,
                                                    BoundingBox<double> const&,
                                                    double);

}  // namespace celeritas

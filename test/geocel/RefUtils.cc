//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/RefUtils.cc
//---------------------------------------------------------------------------//
#include "RefUtils.hh"

#include <cmath>

#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/BoundingBox.hh"

#include "AssertionHelper.hh"
#include "RefComparer.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T>
bool is_ref_eq(AssertionHelper& helper,
               BoundingBox<T> const& expected,
               BoundingBox<T> const& actual)
{
    return is_ref_eq(helper, expected, actual, SoftEqual<T>{}.rel());
}

template<class T>
bool is_ref_eq(AssertionHelper& helper,
               BoundingBox<T> const& expected,
               BoundingBox<T> const& actual,
               T tol)
{
    RefComparer<BoundingBox<T>> compare{helper, expected, actual};
    auto fail = [&] {
        auto&& msg = helper.fail();
        msg << "Expected " << repr(expected) << " but got " << repr(actual);
        msg << " (" << helper.tol_expr() << '=' << tol << ')';
    };

    if (!expected || !actual)
    {
        if (static_cast<bool>(expected) != static_cast<bool>(actual))
        {
            fail();
            compare.record_result(false);
        }
        return static_cast<bool>(compare);
    }

    auto const points_are_close = [tol](auto const& expected_point,
                                        auto const& actual_point) {
        T point_distance{};
        for (size_type i = 0; i != expected_point.size(); ++i)
        {
            if (expected_point[i] == actual_point[i])
            {
                continue;
            }
            if (!std::isfinite(expected_point[i])
                || !std::isfinite(actual_point[i]))
            {
                return false;
            }
            point_distance = std::hypot(point_distance,
                                        expected_point[i] - actual_point[i]);
        }
        return point_distance <= tol;
    };
    bool const result = points_are_close(expected.lower(), actual.lower())
                        && points_are_close(expected.upper(), actual.upper());
    if (!result)
    {
        fail();
    }
    compare.record_result(result);
    return static_cast<bool>(compare);
}

template bool is_ref_eq<float>(
    AssertionHelper&, BoundingBox<float> const&, BoundingBox<float> const&);
template bool is_ref_eq<double>(
    AssertionHelper&, BoundingBox<double> const&, BoundingBox<double> const&);
template bool is_ref_eq<float>(AssertionHelper&,
                               BoundingBox<float> const&,
                               BoundingBox<float> const&,
                               float);
template bool is_ref_eq<double>(AssertionHelper&,
                                BoundingBox<double> const&,
                                BoundingBox<double> const&,
                                double);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

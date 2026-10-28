//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/RefUtils.cc
//---------------------------------------------------------------------------//
#include "RefUtils.hh"

#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/BoundingBox.hh"

#include "AssertionHelper.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T>
bool is_ref_eq(AssertionHelper& helper,
               char const* field_name,
               BoundingBox<T> const& expected,
               BoundingBox<T> const& actual,
               std::nullptr_t)
{
    return is_ref_eq(
        helper, field_name, expected, actual, SoftEqual<T>{}.rel());
}

template<class T>
bool is_ref_eq(AssertionHelper& helper,
               char const* field_name,
               BoundingBox<T> const& expected,
               BoundingBox<T> const& actual,
               T tol)
{
    auto fail = [&] {
        auto&& msg = helper.fail();
        msg << "Expected " << field_name << ": " << repr(expected)
            << " but got " << repr(actual);
        if (helper.tol_expr())
        {
            msg << " (tolerance " << helper.tol_expr() << '=' << tol << ')';
        }
        else
        {
            msg << " (tolerance " << tol << ')';
        }
    };

    if (!expected || !actual)
    {
        if (static_cast<bool>(expected) != static_cast<bool>(actual))
        {
            fail();
            return false;
        }
        return true;
    }

    auto const points_are_close
        = [tol](auto const& expected_point, auto const& actual_point) {
              return expected_point == actual_point
                     || distance(expected_point, actual_point) <= tol;
          };
    if (!points_are_close(expected.lower(), actual.lower())
        || !points_are_close(expected.upper(), actual.upper()))
    {
        fail();
        return false;
    }
    return true;
}

template bool is_ref_eq<float>(AssertionHelper&,
                               char const*,
                               BoundingBox<float> const&,
                               BoundingBox<float> const&,
                               std::nullptr_t);
template bool is_ref_eq<double>(AssertionHelper&,
                                char const*,
                                BoundingBox<double> const&,
                                BoundingBox<double> const&,
                                std::nullptr_t);
template bool is_ref_eq<float>(AssertionHelper&,
                               char const*,
                               BoundingBox<float> const&,
                               BoundingBox<float> const&,
                               float);
template bool is_ref_eq<double>(AssertionHelper&,
                                char const*,
                                BoundingBox<double> const&,
                                BoundingBox<double> const&,
                                double);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

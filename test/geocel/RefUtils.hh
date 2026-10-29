//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/RefUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

namespace celeritas
{
template<class T>
class BoundingBox;

namespace test
{
class AssertionHelper;

template<class T>
bool is_ref_eq(AssertionHelper& helper,
               char const* field_name,
               BoundingBox<T> const& expected,
               BoundingBox<T> const& actual,
               std::nullptr_t);

template<class T>
bool is_ref_eq(AssertionHelper& helper,
               char const* field_name,
               BoundingBox<T> const& expected,
               BoundingBox<T> const& actual,
               T tol);

template<class T>
void check_ref_eq(AssertionHelper& helper,
                  BoundingBox<T> const& expected,
                  BoundingBox<T> const& actual)
{
    (void)is_ref_eq(helper, "BoundingBox", expected, actual, nullptr);
}

template<class T>
void check_ref_eq(AssertionHelper& helper,
                  BoundingBox<T> const& expected,
                  BoundingBox<T> const& actual,
                  T tol)
{
    (void)is_ref_eq(helper, "BoundingBox", expected, actual, tol);
}

extern template bool is_ref_eq<float>(AssertionHelper&,
                                      char const*,
                                      BoundingBox<float> const&,
                                      BoundingBox<float> const&,
                                      std::nullptr_t);
extern template bool is_ref_eq<double>(AssertionHelper&,
                                       char const*,
                                       BoundingBox<double> const&,
                                       BoundingBox<double> const&,
                                       std::nullptr_t);
extern template bool is_ref_eq<float>(AssertionHelper&,
                                      char const*,
                                      BoundingBox<float> const&,
                                      BoundingBox<float> const&,
                                      float);
extern template bool is_ref_eq<double>(AssertionHelper&,
                                       char const*,
                                       BoundingBox<double> const&,
                                       BoundingBox<double> const&,
                                       double);

}  // namespace test
}  // namespace celeritas

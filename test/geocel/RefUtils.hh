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
bool check_ref_eq(AssertionHelper& helper,
                  BoundingBox<T> const& expected,
                  BoundingBox<T> const& actual);

template<class T>
bool check_ref_eq(AssertionHelper& helper,
                  BoundingBox<T> const& expected,
                  BoundingBox<T> const& actual,
                  T tol);

extern template bool check_ref_eq<float>(
    AssertionHelper&, BoundingBox<float> const&, BoundingBox<float> const&);
extern template bool check_ref_eq<double>(
    AssertionHelper&, BoundingBox<double> const&, BoundingBox<double> const&);
extern template bool check_ref_eq<float>(AssertionHelper&,
                                         BoundingBox<float> const&,
                                         BoundingBox<float> const&,
                                         float);
extern template bool check_ref_eq<double>(AssertionHelper&,
                                          BoundingBox<double> const&,
                                          BoundingBox<double> const&,
                                          double);

}  // namespace test
}  // namespace celeritas

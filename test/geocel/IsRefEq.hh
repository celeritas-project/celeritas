//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/IsRefEq.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

namespace celeritas
{
template<class T>
class BoundingBox;

//---------------------------------------------------------------------------//
//! Compare bounding-box corners using an absolute Euclidean distance tolerance
template<class T>
::testing::AssertionResult IsRefEq(char const* expected_expr,
                                   char const* actual_expr,
                                   char const* tol_expr,
                                   BoundingBox<T> const& expected,
                                   BoundingBox<T> const& actual,
                                   T tol);

extern template ::testing::AssertionResult IsRefEq<float>(
    char const*,
    char const*,
    char const*,
    BoundingBox<float> const&,
    BoundingBox<float> const&,
    float);
extern template ::testing::AssertionResult IsRefEq<double>(
    char const*,
    char const*,
    char const*,
    BoundingBox<double> const&,
    BoundingBox<double> const&,
    double);

}  // namespace celeritas

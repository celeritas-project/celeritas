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
namespace test
{
class AssertionHelper;
}

template<class T>
class BoundingBox;

//---------------------------------------------------------------------------//
//! Compare bounding-box corners using an absolute Euclidean distance tolerance
template<class T>
void check_ref_eq(test::AssertionHelper& helper,
                  BoundingBox<T> const& expected,
                  BoundingBox<T> const& actual,
                  T tol);

extern template void check_ref_eq<float>(test::AssertionHelper&,
                                         BoundingBox<float> const&,
                                         BoundingBox<float> const&,
                                         float);
extern template void check_ref_eq<double>(test::AssertionHelper&,
                                          BoundingBox<double> const&,
                                          BoundingBox<double> const&,
                                          double);

}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Comparer.hh
//! \sa TestMacros.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "testdetail/TestMacrosImpl.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Compare selected fields of two reference-result objects.
 */
template<class T>
class Comparer
{
  public:
    Comparer(AssertionHelper& helper, T const& expected, T const& actual)
        : helper_{helper}, expected_{expected}, actual_{actual}
    {
    }

    template<class M>
    bool is_soft_eq(char const* attr, M T::* member) const
    {
        return celeritas::test::is_vec_soft_eq(
            helper_, attr, expected_.*member, actual_.*member);
    }

    template<class M, class Tol>
    bool is_soft_eq(char const* attr, M T::* member, Tol&& tol) const
    {
        return celeritas::test::is_vec_soft_eq(helper_,
                                               attr,
                                               expected_.*member,
                                               actual_.*member,
                                               std::forward<Tol>(tol));
    }

    template<class M>
    bool is_eq(char const* attr, M T::* member) const
    {
        return celeritas::test::is_vec_eq(
            helper_, attr, expected_.*member, actual_.*member);
    }

  private:
    AssertionHelper& helper_;
    T const& expected_;
    T const& actual_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

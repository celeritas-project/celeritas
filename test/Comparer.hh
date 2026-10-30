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
 * Compare selected container fields of two reference-result objects.
 *
 * The comparer stores references to the assertion helper and the expected
 * and actual objects. Each comparison method takes a pointer to a member of
 * the result type; the attribute name is used to identify failures in the
 * assertion output. Exact comparison checks container elements for equality,
 * while soft comparison uses the default soft comparator or a caller-supplied
 * tolerance.
 *
 * \par Example
 *
 * Typically \c CELER_INVOKE_ATTR is used to reduce duplication.
 * \code
    AssertionHelper helper{"expected", "actual"};
    Comparer<MyResult> compare{helper, expected, actual};
    compare.is_eq("track_id", &MyResult::track_id);
    CELER_INVOKE_ATTR(compare.is_eq, MyResult, track_id);
    CELER_INVOKE_ATTR(compare.is_soft_eq, MyResult, position);
    CELER_INVOKE_ATTR(compare.is_soft_eq, MyResult, energy, 1e-5);
 * \endcode
 */
template<class T>
class Comparer
{
  public:
    //! Create a comparer for two results and an assertion helper
    Comparer(AssertionHelper& helper, T const& expected, T const& actual)
        : helper_{helper}, expected_{expected}, actual_{actual}
    {
    }

    //! Compare a member container using the default soft comparator
    template<class M>
    bool is_soft_eq(char const* attr, M T::* member) const
    {
        return celeritas::test::is_vec_soft_eq(
            helper_, attr, expected_.*member, actual_.*member);
    }

    //! Compare a member container using an explicit soft-comparison tolerance
    template<class M, class Tol>
    bool is_soft_eq(char const* attr, M T::* member, Tol&& tol) const
    {
        return celeritas::test::is_vec_soft_eq(helper_,
                                               attr,
                                               expected_.*member,
                                               actual_.*member,
                                               std::forward<Tol>(tol));
    }

    //! Compare a member container using exact element equality
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

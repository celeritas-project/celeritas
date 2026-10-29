//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectTestResult.cc
//---------------------------------------------------------------------------//
#include "IntersectTestResult.hh"

#include <iostream>

#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/RefUtils.hh"

#include "AssertionHelper.hh"
#include "TestMacros.hh"

namespace celeritas
{

template<class T>
struct ReprTraits<BoundingBox<T>>
{
    static void init(std::ostream& os) { ReprTraits<T>::init(os); }

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_container_type<T>(os, "BoundingBox", name);
    }

    static void print_value(std::ostream& os, BoundingBox<T> const& bb)
    {
        os << '{';
        if (bb)
        {
            using ArrayT3 = Array<T, 3>;
            ReprTraits<ArrayT3>::print_value(os, bb.lower());
            os << ", ";
            ReprTraits<ArrayT3>::print_value(os, bb.upper());
        }
        os << '}';
    }
};

namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, IntersectTestResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, node) << ", "
              << CELER_ATTR_REPR(ref, surfaces) << ", "
              << CELER_ATTR_REPR(ref, interior) << ", "
              << CELER_ATTR_REPR(ref, exterior) << '}';
}

//---------------------------------------------------------------------------//
bool check_ref_eq(::celeritas::test::AssertionHelper& helper,
                  IntersectTestResult const& val1,
                  IntersectTestResult const& val2)
{
    using namespace celeritas::test;

    bool result = true;

    result = is_eq(helper, "node", val1.node, val2.node) && result;
    result = is_eq(helper, "surfaces", val1.surfaces, val2.surfaces) && result;
    if (!check_ref_eq(helper, val1.interior, val2.interior))
    {
        helper.fail() << "in interior";
        result = false;
    }
    if (!check_ref_eq(helper, val1.exterior, val2.exterior))
    {
        helper.fail() << "in exterior";
        result = false;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

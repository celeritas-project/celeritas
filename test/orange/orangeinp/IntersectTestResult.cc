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
#include "RefComparer.hh"
#include "TestMacros.hh"

namespace celeritas
{

template<class T>
struct BoundingBoxRepr
{
    BoundingBox<T> const& obj;

    friend std::ostream& operator<<(std::ostream& os,
                                    BoundingBoxRepr const& value)
    {
        os << '{';
        if (value.obj)
        {
            os << repr(value.obj.lower());
            os << ", ";
            os << repr(value.obj.upper());
        }
        return os << '}';
    }
};

template<class T>
BoundingBoxRepr<T> repr(BoundingBox<T> const& bb)
{
    return {bb};
}

namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, IntersectTestResult const& ref)
{
    return os << '{' << CELER_REPR_ATTR(ref, node) << ", "
              << CELER_REPR_ATTR(ref, surfaces) << ", "
              << CELER_REPR_ATTR(ref, interior) << ", "
              << CELER_REPR_ATTR(ref, exterior) << '}';
}

//---------------------------------------------------------------------------//
bool is_ref_eq(::celeritas::test::AssertionHelper& helper,
               IntersectTestResult const& val1,
               IntersectTestResult const& val2)
{
    using namespace celeritas::test;

    RefComparer<IntersectTestResult> compare{helper, val1, val2};
    compare.record_result(is_eq(helper, "node", val1.node, val2.node));
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, IntersectTestResult, surfaces);

    bool const interior_result
        = is_ref_eq(helper, val1.interior, val2.interior);
    if (!interior_result)
    {
        helper.fail() << "in interior";
    }
    compare.record_result(interior_result);

    bool const exterior_result
        = is_ref_eq(helper, val1.exterior, val2.exterior);
    if (!exterior_result)
    {
        helper.fail() << "in exterior";
    }
    compare.record_result(exterior_result);
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

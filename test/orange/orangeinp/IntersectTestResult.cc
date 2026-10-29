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
    return os << '{' << repr(ref.node) << ", " << repr(ref.surfaces) << ", "
              << repr(ref.interior) << ", " << repr(ref.exterior) << '}';
}

//---------------------------------------------------------------------------//
void check_ref_eq(::celeritas::test::AssertionHelper& result,
                  IntersectTestResult const& val1,
                  IntersectTestResult const& val2,
                  std::nullptr_t)
{
#define IRE_COMPARE(ATTR) \
    if (val1.ATTR != val2.ATTR) \
    { \
        result.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR); \
    } \
    else \
        (void)sizeof(char)

    IRE_COMPARE(node);
    IRE_COMPARE(surfaces);

    using celeritas::test::check_ref_eq;
    is_ref_eq(result, "interior", val1.interior, val2.interior, nullptr);
    is_ref_eq(result, "exterior", val1.exterior, val2.exterior, nullptr);
#undef IRE_COMPARE
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

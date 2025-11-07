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

#include "testdetail/TestMacrosImpl.hh"

//!@{
//! Helper macros
#define CELER_REF_ATTR(ATTR) "ref." #ATTR " = " << repr(this->ATTR) << ";\n"
//!@}

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
void IntersectTestResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "IntersectTestResult ref;\n"
         << CELER_REF_ATTR(node) << CELER_REF_ATTR(surfaces)
         << CELER_REF_ATTR(interior) << CELER_REF_ATTR(exterior)
         << "EXPECT_REF_EQ(ref, result);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   IntersectTestResult const& val1,
                                   IntersectTestResult const& val2)
{
    ::celeritas::test::AssertionHelper result{expr1, expr2};

#define IRE_COMPARE(ATTR)                                          \
    if (val1.ATTR != val2.ATTR)                                    \
    {                                                              \
        result.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR);           \
    }                                                              \
    else                                                           \
        (void)sizeof(char)

    IRE_COMPARE(node);
    IRE_COMPARE(surfaces);

    // Special handling for BBox objects with soft equality
    // TODO make its own expect-ref-eq
    auto check_bbox =
        [&result](char const* name, auto const& expected, auto const& actual) {
            if (!expected && !actual)
            {
                return;  // Both null
            }
            if (static_cast<bool>(expected) != static_cast<bool>(actual))
            {
                result.fail() << "Expected " << name << ": " << repr(expected)
                              << " but got " << repr(actual);
                return;
            }

            EqualOr<SoftEqual<>> soft_eq;
            for (int i = 0; i < 3; ++i)
            {
                if (!soft_eq(expected.lower()[i], actual.lower()[i])
                    || !soft_eq(expected.upper()[i], actual.upper()[i]))
                {
                    result.fail()
                        << "Expected " << name << ": " << repr(expected)
                        << " but got " << repr(actual);
                    return;
                }
            }
        };

    check_bbox("interior", val1.interior, val2.interior);
    check_bbox("exterior", val1.exterior, val2.exterior);
#undef IRE_COMPARE
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas

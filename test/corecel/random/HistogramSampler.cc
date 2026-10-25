//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/HistogramSampler.cc
//---------------------------------------------------------------------------//
#include "HistogramSampler.hh"

#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"

#include "AssertionHelper.hh"
#include "testdetail/TestMacrosImpl.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

void SampledHistogram::print_expected() const
{
    std::cout << "SampledHistogram ref;\n"
              << "ref.distribution = " << repr(distribution) << ";\n"
              << "ref.rng_count = " << repr(rng_count) << ";\n"
              << "EXPECT_REF_EQ(ref, result);\n";
}

//---------------------------------------------------------------------------//
/*!
 * Print to a stream.
 */
std::ostream& operator<<(std::ostream& os, SampledHistogram const& sh)
{
    os << "{" << repr(sh.distribution) << ", " << repr(sh.rng_count) << "}";
    return os;
}

//---------------------------------------------------------------------------//

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   SampledHistogram const& val1,
                                   SampledHistogram const& val2)
{
    AssertionHelper result(expr1, expr2);
    result.vec_soft_eq("distribution", val1.distribution, val2.distribution);
    if (!soft_equal(val1.rng_count, val2.rng_count))
    {
        result.fail() << "  rng_count: " << val1.rng_count
                      << " != " << val2.rng_count;
    }

    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

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

//! Print to a stream
std::ostream& operator<<(std::ostream& os, SampledHistogram const& sh)
{
    os << "{" << repr(sh.distribution) << ", " << repr(sh.rng_count) << "}";
    return os;
}

//! Check against a reference histogram
bool check_ref_eq(AssertionHelper& helper,
                  SampledHistogram const& val1,
                  SampledHistogram const& val2,
                  std::nullptr_t)
{
    bool result = is_vec_soft_eq(
        helper, "distribution", val1.distribution, val2.distribution);
    if (!soft_equal(val1.rng_count, val2.rng_count))
    {
        helper.fail() << "  rng_count: " << val1.rng_count
                      << " != " << val2.rng_count;
        result = false;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

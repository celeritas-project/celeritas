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
                  double tol)
{
    bool result = is_vec_soft_eq(
        helper, "distribution", val1.distribution, val2.distribution, tol);
    if (!is_soft_eq(helper, val1.rng_count, val2.rng_count, {tol}))
    {
        return false;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/NumericLimits.test.hh
//---------------------------------------------------------------------------//

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Output results
template<class T>
struct NLTestOutput
{
    T eps;
    T nan;
    T inf;
    T max;
    T inv_zero;  // Test for expected infinity
};

//---------------------------------------------------------------------------//
//! Run on device and return results
template<class T>
NLTestOutput<T> nl_test();

#if !CELER_USE_DEVICE
template<class T>
inline NLTestOutput<T> nl_test()
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

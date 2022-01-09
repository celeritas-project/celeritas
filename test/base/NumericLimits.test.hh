//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NumericLimits.test.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas_test
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
};

//---------------------------------------------------------------------------//
//! Run on device and return results
template<class T>
NLTestOutput<T> nl_test();

#if !CELERITAS_USE_CUDA
template<class T>
inline NLTestOutput<T> nl_test()
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test

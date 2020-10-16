//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NumericLimits.test.hh
//---------------------------------------------------------------------------//

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

//---------------------------------------------------------------------------//
} // namespace celeritas_test

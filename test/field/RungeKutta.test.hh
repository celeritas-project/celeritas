//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FieldTestParams.hh"
#include "base/Types.hh"
#include <vector>

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

#ifdef CELERITAS_USE_CUDA                                                     
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//
//! Output results
struct RK4TestOutput
{
    std::vector<real_type> pos_x;
    std::vector<real_type> mom_y;
    std::vector<real_type> error;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
RK4TestOutput rk4_test(FieldTestParams test_param);

#endif   

//---------------------------------------------------------------------------//
} // namespace celeritas_test

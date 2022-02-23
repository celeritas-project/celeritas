//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas_config.h"
#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

#include "FieldTestParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Output results
struct RK4TestOutput
{
    using real_type = celeritas::real_type;

    std::vector<real_type> pos_x;
    std::vector<real_type> pos_z;
    std::vector<real_type> mom_y;
    std::vector<real_type> mom_z;
    std::vector<real_type> error;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
RK4TestOutput rk4_test(FieldTestParams test_param);

#if !CELER_USE_DEVICE
inline RK4TestOutput rk4_test(FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test

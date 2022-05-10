//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/field/FieldParamsData.hh"

#include "FieldTestParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Output results
struct FITestOutput
{
    std::vector<double> pos_x;
    std::vector<double> pos_z;
    std::vector<double> mom_y;
    std::vector<double> mom_z;
    std::vector<double> error;
};

struct OneGoodStepOutput
{
    std::vector<double> pos_x;
    std::vector<double> pos_z;
    std::vector<double> mom_y;
    std::vector<double> mom_z;
    std::vector<double> length;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
FITestOutput
driver_test(const celeritas::FieldParamsData& fd_ptr, FieldTestParams tp);

OneGoodStepOutput
accurate_advance_test(const celeritas::FieldParamsData& fd_ptr,
                      FieldTestParams                   tp);

#if !CELER_USE_DEVICE
inline FITestOutput
driver_test(const celeritas::FieldParamsData&, FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline OneGoodStepOutput
accurate_advance_test(const celeritas::FieldParamsData&, FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test

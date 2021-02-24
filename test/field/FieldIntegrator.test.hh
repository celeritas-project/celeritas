//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldDriver.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FieldTestParams.hh"
#include "field/FieldParamsPointers.hh"
#include <vector>

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! Output results
struct FDTestOutput
{
    std::vector<double> pos;
    std::vector<double> mom;
    std::vector<double> err;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
FDTestOutput integrator_test(const celeritas::FieldParamsPointers& fd_ptr,
                             FieldTestParams tp);

//---------------------------------------------------------------------------//
} // namespace celeritas_test

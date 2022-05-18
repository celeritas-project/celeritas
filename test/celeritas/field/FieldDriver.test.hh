//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MagFieldEquation.hh"

#include "FieldTestParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<template<class EquationT> class StepperT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_driver(FieldT&&                             field,
                      const celeritas::FieldDriverOptions& options,
                      celeritas::units::ElementaryCharge   charge)
{
    using Equation_t = celeritas::MagFieldEquation<FieldT>;
    using Stepper_t  = StepperT<Equation_t>;
    using Driver_t   = celeritas::FieldDriver<Stepper_t>;
    return Driver_t{
        options,
        Stepper_t{Equation_t{::celeritas::forward<FieldT>(field), charge}}};
}

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
driver_test(const celeritas::FieldDriverOptions& fd_ptr, FieldTestParams tp);

OneGoodStepOutput
accurate_advance_test(const celeritas::FieldDriverOptions& fd_ptr,
                      FieldTestParams                      tp);

#if !CELER_USE_DEVICE
inline FITestOutput
driver_test(const celeritas::FieldDriverOptions&, FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline OneGoodStepOutput
accurate_advance_test(const celeritas::FieldDriverOptions&, FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test

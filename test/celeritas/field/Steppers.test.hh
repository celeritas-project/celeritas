//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Steppers.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/field/MagFieldEquation.hh"

#include "FieldTestParams.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<template<class EquationT> class StepperT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_stepper(FieldT&&                           field,
                       celeritas::units::ElementaryCharge charge)
{
    using Equation_t = celeritas::MagFieldEquation<FieldT>;
    using Stepper_t  = StepperT<Equation_t>;
    return Stepper_t{Equation_t{::celeritas::forward<FieldT>(field), charge}};
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Output results
struct StepperTestOutput
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
StepperTestOutput helix_test(FieldTestParams test_param);
StepperTestOutput rk4_test(FieldTestParams test_param);
StepperTestOutput dp547_test(FieldTestParams test_param);

#if !CELER_USE_DEVICE
inline StepperTestOutput helix_test(FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline StepperTestOutput rk4_test(FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline StepperTestOutput dp547_test(FieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test

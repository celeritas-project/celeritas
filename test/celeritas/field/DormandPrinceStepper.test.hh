//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/Logger.hh"
#include "celeritas/field/DormandPrinceMultiStepperGlobal.cuda.hh"
#include "celeritas/field/DormandPrinceMultiStepperShared.cuda.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using celeritas::units::ElementaryCharge;
using EvaluatorType = celeritas::MagFieldEquation<Real3 (&)(Real3 const&)>;
using StepperUni = celeritas::DormandPrinceStepper<EvaluatorType&>;
using StepperMultiGlobal
    = celeritas::DormandPrinceMultiStepperGlobal<EvaluatorType&>;
using StepperMultiShared
    = celeritas::DormandPrinceMultiStepperShared<EvaluatorType&>;

//---------------------------------------------------------------------------//
// CONSTANTS
//---------------------------------------------------------------------------//
constexpr int one_thread = 1;
constexpr int multi_thread = 4;
constexpr int number_iterations = 40;
constexpr OdeState initial_states[5] = {
    OdeState{{1, 2, 3}, {1, 1, 1}},
    OdeState{{0, 0, 0}, {0, 0, 1}},
    OdeState{{-1, -2, -3}, {0, 0, 1}},
    OdeState{{1, 2, 3}, {0, 0, -1}},
    OdeState{{1, 2, 3}, {0, 5, 0}},
};
constexpr int number_states_sample = sizeof(initial_states)
                                     / sizeof(initial_states[0]);

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
inline CELER_FUNCTION Real3 dormand_prince_dummy_field(Real3 const& pos)
{
    Real3 result;
    result[0] = real_type(0.5) * pos[1];
    result[1] = real_type(1.0) * pos[2];
    result[2] = real_type(2.0) * pos[0];
    return result;
}

template<class FieldT>
inline CELER_FUNCTION decltype(auto) make_dummy_equation(FieldT&& field)
{
    using Equation_t = celeritas::MagFieldEquation<FieldT>;

    return Equation_t{::celeritas::forward<FieldT>(field), ElementaryCharge{3}};
}

inline void build_variables(int number_states,
                            bool global_version,
                            FieldStepperResult* results,
                            OdeState* along_states,
                            OdeState* states)
{
    for (int i = 0; i < number_states; ++i)
    {
        results[i] = FieldStepperResult();
        states[i] = initial_states[i % number_states_sample];
        if (global_version){
            along_states[i] = OdeState();
        }
    }
}


struct KernelResult
{
    FieldStepperResult* results;
    float milliseconds;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
KernelResult simulate_multi_next_chord(int number_threads,
                                       int number_states,
                                       bool use_shared = false);

#if !CELER_USE_DEVICE
inline KernelResult simulate_multi_next_chord(int number_threads,
                                              int number_states,
                                              bool use_shared)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
    CELER_LOG(error) << "Cannot simulate multi next chord with those "
                        "settings\tnumber_threads: "
                     << number_threads << "\tnumber_states: " << number_states
                     << "\tuse_shared: " << use_shared;
    return KernelResult{};
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

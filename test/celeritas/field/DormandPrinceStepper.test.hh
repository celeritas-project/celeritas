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

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using celeritas::units::ElementaryCharge;
using Evaluator_t = celeritas::MagFieldEquation<Real3 (&)(Real3 const&)>;
using Stepper_uni = celeritas::DormandPrinceStepper<Evaluator_t&>;
using Stepper_multi_global
    = celeritas::DormandPrinceMultiStepperGlobal<Evaluator_t&>;
using Stepper_multi_shared
    = celeritas::DormandPrinceMultiStepperShared<Evaluator_t&>;

//---------------------------------------------------------------------------//
// CONSTANTS
//---------------------------------------------------------------------------//
constexpr int one_thread = 1;
constexpr int multi_thread = 4;
constexpr int number_iterations = 1;
constexpr OdeState initial_states[5] = {
    OdeState{{1, 2, 3}, {1, 1, 1}},
    OdeState{{0, 0, 0}, {0, 0, 1}},
    OdeState{{-1, -2, -3}, {0, 0, 1}},
    OdeState{{1, 2, 3}, {0, 0, -1}},
    OdeState{{1, 2, 3}, {0, 5, 0}},
};
constexpr int number_of_states = sizeof(initial_states)
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

inline void print_result(FieldStepperResult const& result)
{
    CELER_LOG(info)
        << "Final mid state position:   " << result.mid_state.pos[0] << ", "
        << result.mid_state.pos[1] << ", " << result.mid_state.pos[2];
    CELER_LOG(info)
        << "Final mid state momentum:   " << result.mid_state.mom[0] << ", "
        << result.mid_state.mom[1] << ", " << result.mid_state.mom[2];
    CELER_LOG(info)
        << "Final error state position: " << result.err_state.pos[0] << ", "
        << result.err_state.pos[1] << ", " << result.err_state.pos[2];
    CELER_LOG(info)
        << "Final error state momentum: " << result.err_state.mom[0] << ", "
        << result.err_state.mom[1] << ", " << result.err_state.mom[2];
    CELER_LOG(info)
        << "Final end state position:   " << result.end_state.pos[0] << ", "
        << result.end_state.pos[1] << ", " << result.end_state.pos[2];
    CELER_LOG(info)
        << "Final end state momentum:   " << result.end_state.mom[0] << ", "
        << result.end_state.mom[1] << ", " << result.end_state.mom[2];
}

inline std::string print_results(FieldStepperResult const& expected,
                                 FieldStepperResult const& actual)
{
    std::string result;
    result = "Expected mid state position:   "
             + std::to_string(expected.mid_state.pos[0]) + ", "
             + std::to_string(expected.mid_state.pos[1]) + ", "
             + std::to_string(expected.mid_state.pos[2]) + "\n";
    result += "Actual mid state position:     "
              + std::to_string(actual.mid_state.pos[0]) + ", "
              + std::to_string(actual.mid_state.pos[1]) + ", "
              + std::to_string(actual.mid_state.pos[2]) + "\n";
    result += "Expected mid state momentum:   "
              + std::to_string(expected.mid_state.mom[0]) + ", "
              + std::to_string(expected.mid_state.mom[1]) + ", "
              + std::to_string(expected.mid_state.mom[2]) + "\n";
    result += "Actual mid state momentum:     "
              + std::to_string(actual.mid_state.mom[0]) + ", "
              + std::to_string(actual.mid_state.mom[1]) + ", "
              + std::to_string(actual.mid_state.mom[2]) + "\n";
    result += "Expected error state position: "
              + std::to_string(expected.err_state.pos[0]) + ", "
              + std::to_string(expected.err_state.pos[1]) + ", "
              + std::to_string(expected.err_state.pos[2]) + "\n";
    result += "Actual error state position:   "
              + std::to_string(actual.err_state.pos[0]) + ", "
              + std::to_string(actual.err_state.pos[1]) + ", "
              + std::to_string(actual.err_state.pos[2]) + "\n";
    result += "Expected error state momentum: "
              + std::to_string(expected.err_state.mom[0]) + ", "
              + std::to_string(expected.err_state.mom[1]) + ", "
              + std::to_string(expected.err_state.mom[2]) + "\n";
    result += "Actual error state momentum:   "
              + std::to_string(actual.err_state.mom[0]) + ", "
              + std::to_string(actual.err_state.mom[1]) + ", "
              + std::to_string(actual.err_state.mom[2]) + "\n";
    result += "Expected end state position:   "
              + std::to_string(expected.end_state.pos[0]) + ", "
              + std::to_string(expected.end_state.pos[1]) + ", "
              + std::to_string(expected.end_state.pos[2]) + "\n";
    result += "Actual end state position:     "
              + std::to_string(actual.end_state.pos[0]) + ", "
              + std::to_string(actual.end_state.pos[1]) + ", "
              + std::to_string(actual.end_state.pos[2]) + "\n";
    result += "Expected end state momentum:   "
              + std::to_string(expected.end_state.mom[0]) + ", "
              + std::to_string(expected.end_state.mom[1]) + ", "
              + std::to_string(expected.end_state.mom[2]) + "\n";
    result += "Actual end state momentum:     "
              + std::to_string(actual.end_state.mom[0]) + ", "
              + std::to_string(actual.end_state.mom[1]) + ", "
              + std::to_string(actual.end_state.mom[2]) + "\n";
    return result;
}

inline CELER_FUNCTION bool
compare_results(FieldStepperResult& e1, FieldStepperResult& e2)
{
    // Check that the results isn't 0
    if (e1.mid_state.pos[0] == 0 && e1.mid_state.pos[1] == 0
        && e1.mid_state.pos[2] == 0 && e1.mid_state.mom[0] == 0
        && e1.mid_state.mom[1] == 0 && e1.mid_state.mom[2] == 0
        && e1.err_state.pos[0] == 0 && e1.err_state.pos[1] == 0
        && e1.err_state.pos[2] == 0 && e1.err_state.mom[0] == 0
        && e1.err_state.mom[1] == 0 && e1.err_state.mom[2] == 0
        && e1.end_state.pos[0] == 0 && e1.end_state.pos[1] == 0
        && e1.end_state.pos[2] == 0 && e1.end_state.mom[0] == 0
        && e1.end_state.mom[1] == 0 && e1.end_state.mom[2] == 0)
        return false;

    // Comparing mid state
    if (e1.mid_state.pos[0] != e2.mid_state.pos[0])
        return false;
    if (e1.mid_state.pos[1] != e2.mid_state.pos[1])
        return false;
    if (e1.mid_state.pos[2] != e2.mid_state.pos[2])
        return false;
    if (e1.mid_state.mom[0] != e2.mid_state.mom[0])
        return false;
    if (e1.mid_state.mom[1] != e2.mid_state.mom[1])
        return false;
    if (e1.mid_state.mom[2] != e2.mid_state.mom[2])
        return false;

    // Comparing err state
    if (e1.err_state.pos[0] != e2.err_state.pos[0])
        return false;
    if (e1.err_state.pos[1] != e2.err_state.pos[1])
        return false;
    if (e1.err_state.pos[2] != e2.err_state.pos[2])
        return false;
    if (e1.err_state.mom[0] != e2.err_state.mom[0])
        return false;
    if (e1.err_state.mom[1] != e2.err_state.mom[1])
        return false;
    if (e1.err_state.mom[2] != e2.err_state.mom[2])
        return false;

    // Comparing end state
    if (e1.end_state.pos[0] != e2.end_state.pos[0])
        return false;
    if (e1.end_state.pos[1] != e2.end_state.pos[1])
        return false;
    if (e1.end_state.pos[2] != e2.end_state.pos[2])
        return false;
    if (e1.end_state.mom[0] != e2.end_state.mom[0])
        return false;
    if (e1.end_state.mom[1] != e2.end_state.mom[1])
        return false;
    if (e1.end_state.mom[2] != e2.end_state.mom[2])
        return false;

    return true;
}

struct KernelResult
{
    FieldStepperResult* results;
    float milliseconds;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
KernelResult
simulate_multi_next_chord(int number_threads, bool use_shared = false);

#if !CELER_USE_DEVICE
inline KernelResult
simulate_multi_next_chord(int number_threads, bool use_shared = false)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
    return nullptr;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

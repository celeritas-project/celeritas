//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/DormandPrinceStepper.hh" // for DormandPrinceStepper
#include "celeritas/field/MagFieldEquation.hh"    // for MagFieldEquation
#include "corecel/io/Logger.hh" // for CELER_LOG

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using celeritas::units::ElementaryCharge;
// using time_unit = std::chrono::nanoseconds;

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
class DormandPrinceKernelArguments {
    public:
    DormandPrinceKernelArguments(int number_iterations, FieldStepperResult result) : number_iterations(number_iterations), result(result){}
    int number_iterations;
    FieldStepperResult& result;
};

inline CELER_FUNCTION Real3 dormand_prince_dummy_field(Real3 const& pos){
    Real3 result;
    result[0] = real_type(0.5) * pos[1];
    result[1] = real_type(1.0) * pos[2];
    result[2] = real_type(2.0) * pos[0];
    return result;
}

template<class FieldT>
inline CELER_FUNCTION decltype(auto) make_dummy_equation(FieldT&& field){
    using Equation_t = celeritas::MagFieldEquation<FieldT>;

    return Equation_t{::celeritas::forward<FieldT>(field), ElementaryCharge{3}};
}

inline void print_result(FieldStepperResult const& result){
    CELER_LOG(info) << "Final mid state position:   " << result.mid_state.pos[0] << ", " << result.mid_state.pos[1] << ", " << result.mid_state.pos[2];
    CELER_LOG(info) << "Final mid state momentum:   " << result.mid_state.mom[0] << ", " << result.mid_state.mom[1] << ", " << result.mid_state.mom[2];
    CELER_LOG(info) << "Final error state position: " << result.err_state.pos[0] << ", " << result.err_state.pos[1] << ", " << result.err_state.pos[2];
    CELER_LOG(info) << "Final error state momentum: " << result.err_state.mom[0] << ", " << result.err_state.mom[1] << ", " << result.err_state.mom[2];
    CELER_LOG(info) << "Final end state position:   " << result.end_state.pos[0] << ", " << result.end_state.pos[1] << ", " << result.end_state.pos[2];
    CELER_LOG(info) << "Final end state momentum:   " << result.end_state.mom[0] << ", " << result.end_state.mom[1] << ", " << result.end_state.mom[2];
}

inline CELER_FUNCTION void run_one_dormand_prince_step(){

    // Set up equation
    auto eval = make_dummy_equation(dormand_prince_dummy_field);
    auto stepper = DormandPrinceStepper{eval};

    // Set up dummy initial state
    OdeState state = eval({{1, 2, 3}, {0, 0, 1}});

    // set step
    using real_type = double;
    real_type step = 1.0;
    FieldStepperResult result;

    for (int i = 0; i < 50; ++i)
    {
        result = stepper(step, state);
        real_type dchord = detail::distance_chord(
            state, result.mid_state, result.end_state);
        step *= max(std::sqrt(1 / dchord), 0.5);
    }

    // CELER_LOG(debug) << "Initial state position: " << state.pos[0] << ", " << state.pos[1] << ", " << state.pos[2];
    // CELER_LOG(debug) << "Initial state momentum: " << state.mom[0] << ", " << state.mom[1] << ", " << state.mom[2];
    
    // Catch start time
    // auto start = std::chrono::high_resolution_clock::now();
    
    // auto new_state = stepper(1, state);

    // Catch end time
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<time_unit>(end - start);

    // Show result
    // CELER_LOG(debug) << "Final mid state position: " << new_state.mid_state.pos[0] << ", " << new_state.mid_state.pos[1] << ", " << new_state.mid_state.pos[2];
    // CELER_LOG(debug) << "Final mid state momentum: " << new_state.mid_state.mom[0] << ", " << new_state.mid_state.mom[1] << ", " << new_state.mid_state.mom[2];
    // CELER_LOG(debug) << "Final error state position: " << new_state.err_state.pos[0] << ", " << new_state.err_state.pos[1] << ", " << new_state.err_state.pos[2];
    // CELER_LOG(debug) << "Final error state momentum: " << new_state.err_state.mom[0] << ", " << new_state.err_state.mom[1] << ", " << new_state.err_state.mom[2];
    // CELER_LOG(debug) << "Final end state position: " << new_state.end_state.pos[0] << ", " << new_state.end_state.pos[1] << ", " << new_state.end_state.pos[2];
    // CELER_LOG(debug) << "Final end state momentum: " << new_state.end_state.mom[0] << ", " << new_state.end_state.mom[1] << ", " << new_state.end_state.mom[2];

    // Check that the duration is less than 1 second
    // auto time_limit = std::chrono::seconds{1};
    // EXPECT_LT(duration.count(),
    //  std::chrono::duration_cast<time_unit>(time_limit).count());

    // return duration;

}

//---------------------------------------------------------------------------//
//! Run on device and return results
void dormand_prince_cuda_test();

#if !CELER_USE_DEVICE
inline void dormand_prince_cuda_test()
{ 
    CELER_NOT_CONFIGURED("CUDA or HIP");
    return nullptr;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

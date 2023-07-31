//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandMultiPrinceStepperGlobal.cuda.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Based on the DormandPrinceStepper.hh, but with multiple threads.
 */
template<class EquationT>
class DormandPrinceMultiStepperGlobal
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = FieldStepperResult;
    //!@}

  public:
    //! Construct with the equation of motion
    explicit CELER_FUNCTION DormandPrinceMultiStepperGlobal(EquationT&& eq)
        : calc_rhs_(::celeritas::forward<EquationT>(eq))
    {
    }

    // Adaptive step size control
    CELER_FUNCTION result_type operator()(real_type step,
                                          OdeState const& beg_state,
                                          int number_threads,
                                          OdeState* ks,
                                          OdeState* along_state,
                                          FieldStepperResult* result) const;

    CELER_FUNCTION void run_sequential(real_type step,
                                       OdeState const& beg_state,
                                       int id,
                                       int mask,
                                       OdeState* ks,
                                       OdeState* along_state,
                                       FieldStepperResult* result) const;

    CELER_FUNCTION void run_aside(real_type step,
                                  OdeState const& beg_state,
                                  int id,
                                  int index,
                                  int mask,
                                  OdeState* ks,
                                  OdeState* along_state,
                                  FieldStepperResult* result) const;

  private:
    // Functor to calculate the force applied to a particle
    EquationT calc_rhs_;
};

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
#define UPDATE_STATE(index, state, coefficient, k)        \
    state.pos[index - 1] = coefficient * k.pos[index - 1] \
                           + state.pos[index - 1];        \
    state.mom[index - 1] = coefficient * k.mom[index - 1] \
                           + state.mom[index - 1];

#define DISPATCH_VECT_MULT(mask) \
    __syncwarp(mask);            \
    __syncwarp(mask);

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EquationT>
CELER_FUNCTION DormandPrinceMultiStepperGlobal(EquationT&&)
    ->DormandPrinceMultiStepperGlobal<EquationT>;

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE

// TODO: implement this for CPU

#endif  // !CELER_USE_DEVICE
#ifdef __CUDA_ARCH__

//---------------------------------------------------------------------------//
// GPU FUNCTIONS
//---------------------------------------------------------------------------//
template<class E>
inline CELER_FUNCTION auto
DormandPrinceMultiStepperGlobal<E>::operator()(real_type step,
                                               OdeState const& beg_state,
                                               int number_threads,
                                               OdeState* ks,
                                               OdeState* along_state,
                                               FieldStepperResult* result) const
    -> result_type
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x) / number_threads;
    int index = (threadIdx.x + blockIdx.x * blockDim.x) % number_threads;

    int mask = (4 * 4 - 1) << ((id * 4) % 32);

    if (index == 0)
    {
        run_sequential(step, beg_state, id, mask, ks, along_state, result);
    }
    else
    {
        run_aside(step, beg_state, id, index, mask, ks, along_state, result);
    }

    return *result;
}

template<class E>
inline CELER_FUNCTION void
DormandPrinceMultiStepperGlobal<E>::run_aside(real_type step,
                                              OdeState const& beg_state,
                                              int id,
                                              int index,
                                              int mask,
                                              OdeState* ks,
                                              OdeState* along_state,
                                              FieldStepperResult* result) const
{
    using R = real_type;
    // Coefficients for Dormand-Prince Rks[4](4)7M
    constexpr R a11 = 0.2;

    constexpr R a21 = 0.075;
    constexpr R a22 = 0.225;

    constexpr R a31 = 44 / R(45);
    constexpr R a32 = -56 / R(15);
    constexpr R a33 = 32 / R(9);

    constexpr R a41 = 19372 / R(6561);
    constexpr R a42 = -25360 / R(2187);
    constexpr R a43 = 64448 / R(6561);
    constexpr R a44 = -212 / R(729);

    constexpr R a51 = 9017 / R(3168);
    constexpr R a52 = -355 / R(33);
    constexpr R a53 = 46732 / R(5247);
    constexpr R a54 = 49 / R(176);
    constexpr R a55 = -5103 / R(18656);

    constexpr R a61 = 35 / R(384);
    constexpr R a63 = 500 / R(1113);
    constexpr R a64 = 125 / R(192);
    constexpr R a65 = -2187 / R(6784);
    constexpr R a66 = 11 / R(84);

    constexpr R d71 = a61 - 5179 / R(57600);
    constexpr R d73 = a63 - 7571 / R(16695);
    constexpr R d74 = a64 - 393 / R(640);
    constexpr R d75 = a65 + 92097 / R(339200);
    constexpr R d76 = a66 - 187 / R(2100);
    constexpr R d77 = -1 / R(40);

    // Coefficients for the mid point calculation by Shampine
    constexpr R c71 = 6025192743 / R(30085553152);
    constexpr R c73 = 51252292925 / R(65400821598);
    constexpr R c74 = -2691868925 / R(45128329728);
    constexpr R c75 = 187940372067 / R(1594534317056);
    constexpr R c76 = -1776094331 / R(19743644256);
    constexpr R c77 = 11237099 / R(235043384);

    // Coefficients for the vector multiplication
    constexpr R axx[] = {a11, a21, a22, a31, a32, a33, a41, a42, a43, a44,
                         a51, a52, a53, a54, a55, a61, a63, a64, a65, a66};
    constexpr R dxx[] = {d71, d73, d74, d75, d76, d77};
    constexpr R cxx[] = {c71, c73, c74, c75, c76, c77};

    // Vector multiplication for step one to five
    int coef_counter = 0;
    for (int i = 0; i < 5; i++)
    {
        __syncwarp(mask);
        for (int j = 0; j <= i; j++)
        {
            UPDATE_STATE(
                index, (*along_state), step * axx[coef_counter], ks[j]);
            coef_counter++;
        }
        __syncwarp(mask);
    }

    // Vector multiplication for step six: end state
    __syncwarp(mask);
    for (int j = 0; j < 6; j++)
    {
        if (j == 1)
            continue;  // because a62 = 0
        UPDATE_STATE(index, result->end_state, step * axx[coef_counter], ks[j]);
        coef_counter++;
    }
    __syncwarp(mask);

    // Vector mutltiplication for step eight and nine: error and mid state
    __syncwarp(mask);
    coef_counter = 0;
    for (int j = 0; j < 7; j++)
    {
        if (j == 1)
            continue;  // because d72 and c72 = 0
        UPDATE_STATE(index, result->err_state, step * dxx[coef_counter], ks[j]);
        UPDATE_STATE(
            index, result->mid_state, step * cxx[coef_counter] / R(2), ks[j]);
        coef_counter++;
    }
    __syncwarp(mask);
}

template<class E>
inline CELER_FUNCTION void
DormandPrinceMultiStepperGlobal<E>::run_sequential(real_type step,
                                                   OdeState const& beg_state,
                                                   int id,
                                                   int mask,
                                                   OdeState* ks,
                                                   OdeState* along_state,
                                                   FieldStepperResult* result) const
{
    // First step
    ks[0] = calc_rhs_(beg_state);
    *along_state = beg_state;
    DISPATCH_VECT_MULT(mask);

    // Second step
    ks[1] = calc_rhs_(*along_state);
    *along_state = beg_state;
    DISPATCH_VECT_MULT(mask);

    // Third step
    ks[2] = calc_rhs_(*along_state);
    *along_state = beg_state;
    DISPATCH_VECT_MULT(mask);

    // Fourth step
    ks[3] = calc_rhs_(*along_state);
    *along_state = beg_state;
    DISPATCH_VECT_MULT(mask);

    // Fifth step
    ks[4] = calc_rhs_(*along_state);
    *along_state = beg_state;
    DISPATCH_VECT_MULT(mask);

    // Sixth step
    ks[5] = calc_rhs_(*along_state);
    result->end_state = beg_state;
    DISPATCH_VECT_MULT(mask);

    // Seventh step: the final step
    ks[6] = calc_rhs_(result->end_state);

    // The error estimate and the mid point
    result->err_state = {{0, 0, 0}, {0, 0, 0}};
    result->mid_state = beg_state;
    DISPATCH_VECT_MULT(mask);
}

#endif  // __CUDA_ARCH__

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandMultiPrinceStepper.hh
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
class DormandPrinceMultiStepper
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = FieldStepperResult;
    //!@}

  public:
    //! Construct with the equation of motion
    explicit CELER_FUNCTION DormandPrinceMultiStepper(EquationT&& eq)
        : calc_rhs_(::celeritas::forward<EquationT>(eq))
    {
    }

    // Adaptive step size control
    CELER_FUNCTION result_type operator()(real_type step,
                                          OdeState const& beg_state,
                                          int id, int index,
                                          OdeState *ks, OdeState *along_state,
                                          FieldStepperResult *result) const;

    CELER_FUNCTION void run_sequential(real_type step,
                                          OdeState const& beg_state,
                                          int id, int mask,
                                          OdeState *ks, OdeState *along_state,
                                          FieldStepperResult *result) const;

    CELER_FUNCTION void run_aside(real_type step,
                                  OdeState const& beg_state,
                                  int id, int index, int mask,
                                  OdeState *ks, OdeState *along_state,
                                  FieldStepperResult *result) const;

  private:
    // Functor to calculate the force applied to a particle
    EquationT calc_rhs_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EquationT>
CELER_FUNCTION DormandPrinceMultiStepper(EquationT&&)->DormandPrinceMultiStepper<EquationT>;

//---------------------------------------------------------------------------//
}  // namespace celeritas

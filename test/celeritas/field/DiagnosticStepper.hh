//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DiagnosticStepper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <utility>

#include "celeritas/field/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Count the number of invocations to the field stepper.
 *
 * This helps diagnose how many times the field driver advances a step.
 */
template<class StepperT>
class DiagnosticStepper
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = typename StepperT::result_type;
    using size_type = std::size_t;
    //!@}

  public:
    //! Forward construction arguments to the original stepper
    template<class... Args>
    DiagnosticStepper(Args&&... args) : do_step_(std::forward<Args>(args)...)
    {
    }

    //! Calculate a step and increment the counter
    result_type operator()(real_type step, OdeState const& beg_state) const
    {
        ++count_;
        return do_step_(step, beg_state);
    }

    //! Get the number of steps
    size_type count() const { return count_; }
    //! Reset the stepscounter
    void reset_count() { count_ = 0; }

  private:
    StepperT do_step_;
    mutable size_type count_ = 0;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class StepperT>
CELER_FUNCTION DiagnosticStepper(StepperT&&) -> DiagnosticStepper<StepperT>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

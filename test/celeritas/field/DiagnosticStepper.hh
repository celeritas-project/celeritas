//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
    //! Type aliases
    using result_type = typename StepperT::result_type;
    using size_type   = std::size_t;
    //!@}

  public:
    //! Forward construction arguments to the original engine
    template<class... Args>
    DiagnosticStepper(Args&&... args) : do_step_(std::forward<Args>(args)...)
    {
    }

    //! Get a random number and increment the sample counter
    result_type operator()(real_type step, const OdeState& beg_state) const
    {
        ++count_;
        return do_step_(step, beg_state);
    }

    //! Get the number of samples
    size_type count() const { return count_; }
    //! Reset the sample counter
    void reset_count() { count_ = 0; }

  private:
    StepperT          do_step_;
    mutable size_type count_ = 0;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

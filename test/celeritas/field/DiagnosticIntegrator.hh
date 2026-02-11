//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DiagnosticIntegrator.hh
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
 * Count the number of invocations to the field integrator.
 *
 * This helps diagnose how many times the field driver advances a step.
 */
template<class IntegratorT>
class DiagnosticIntegrator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = typename IntegratorT::result_type;
    using size_type = std::size_t;
    //!@}

  public:
    //! Forward construction arguments to the original integrator
    template<class... Args>
    DiagnosticIntegrator(Args&&... args)
        : do_step_(std::forward<Args>(args)...)
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
    //! Get and reset the counter
    size_type exchange_count() { return std::exchange(count_, 0); }

  private:
    IntegratorT do_step_;
    mutable size_type count_ = 0;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class IntegratorT>
CELER_FUNCTION DiagnosticIntegrator(IntegratorT&&)
    -> DiagnosticIntegrator<IntegratorT>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

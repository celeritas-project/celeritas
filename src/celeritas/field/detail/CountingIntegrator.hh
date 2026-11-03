//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/CountingIntegrator.hh
//! \sa test/celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "../FieldPropagationCounters.hh"
#include "../Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Count the number of calls to a field integrator.
 *
 * This is a device-compatible counterpart to the \c DiagnosticIntegrator test
 * helper that increments \c FieldPropagationCounters::integrations.
 */
template<class IntegratorT>
class CountingIntegrator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type =
        typename std::remove_reference_t<IntegratorT>::result_type;
    //!@}

  public:
    //! Construct with the integrator and the counters to increment
    CELER_FUNCTION CountingIntegrator(IntegratorT&& integrate,
                                      FieldPropagationCounters* counters)
        : integrate_{::celeritas::forward<IntegratorT>(integrate)}
        , c_{counters}
    {
        CELER_EXPECT(c_);
    }

    //! Integrate a step and increment the counter
    CELER_FUNCTION result_type operator()(real_type step,
                                          OdeState const& beg_state) const
    {
        ++c_->integrations;
        return integrate_(step, beg_state);
    }

  private:
    IntegratorT integrate_;
    FieldPropagationCounters* c_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class IntegratorT>
CELER_CTAD_FUNCTION CountingIntegrator(IntegratorT&&, FieldPropagationCounters*)
    -> CountingIntegrator<IntegratorT>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

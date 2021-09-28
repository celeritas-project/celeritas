//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "random/distributions/IsotropicDistribution.hh"
#include "AtomicRelaxation.hh"
#include "AtomicRelaxationInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for atomic relaxation.
 *
 * This class can be used inside an \c Interactor to simplify the creation of
 * the sampling distribution for relaxation and the allocation of storage for
 * secondaries created in both relaxation and the primary process.
 *
 * \code
    // Allocate secondaries for a model that produces a single secondary
    Span<Secondary> secondaries;
    size_type       count = relax_helper.max_secondaries() + 1;
    if (Secondary* ptr = allocate_secondaries(count))
    {
        secondaries = {ptr, count};
    }
    else
    {
        return Interaction::from_failure();
    }
    Interaction result;

    // ...

    AtomicRelaxation sample_relaxation = relax_helper.build_distribution(
        cutoffs, shell_id, secondaries.subspan(1));
    auto outgoing             = sample_relaxation(rng);
    result.secondaries        = outgoing.secondaries;
    result.energy_deposition -= outgoing.energy;
   \endcode
 */
class AtomicRelaxationHelper
{
  public:
    // Construct with the currently interacting element
    inline CELER_FUNCTION
    AtomicRelaxationHelper(const AtomicRelaxParamsPointers& shared,
                           const AtomicRelaxStatePointers&  states,
                           ElementId                        el_id,
                           ThreadId                         tid);

    // Whether atomic relaxation should be applied
    explicit inline CELER_FUNCTION operator bool() const;

    // Space needed for relaxation secondaries
    inline CELER_FUNCTION size_type max_secondaries() const;

    // Storage for subshell ID stack
    inline CELER_FUNCTION Span<SubshellId> scratch() const;

    // Create the sampling distribution from sampled shell and allocated mem
    inline CELER_FUNCTION AtomicRelaxation
    build_distribution(const CutoffView& cutoffs,
                       SubshellId        shell_id,
                       Span<Secondary>   secondaries) const;

  private:
    const AtomicRelaxParamsPointers& shared_;
    const AtomicRelaxStatePointers&  states_;
    const ElementId                  el_id_;
    const ThreadId                   thread_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "AtomicRelaxationHelper.i.hh"

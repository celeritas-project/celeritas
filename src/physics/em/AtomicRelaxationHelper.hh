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
    // Construct the helper for a model that produces a single secondary
    AtomicRelaxationHelper relax_helper(shared, el_id, allocate, 1);
    Span<Secondary>  secondaries = relax_helper.allocate_secondaries();
    Span<SubshellId> vacancies   = relax_helper.allocate_vacancies();
    if (secondaries.empty() || (secondaries.size() > 1 && vacancies.empty()))
    {
        return Interaction::from_failure();
    }
    Interaction result;

    // ...

    AtomicRelaxation sample_relaxation
        = relax_helper.build_distribution(secondaries, vacancies, shell_id);
    auto outgoing             = sample_relaxation(rng);
    result.secondaries        = outgoing.secondaries;
    result.energy_deposition -= outgoing.energy;
   \endcode
 *
 * If atomic relaxation is not enabled or does not apply to this element or
 * subshell, the helper will simply allocate space for secondaries created in
 * the primary process and no additional secondaries will be produced in
 * relaxation.
 */
class AtomicRelaxationHelper
{
  public:
    // Construct with the currently interacting element
    inline CELER_FUNCTION
    AtomicRelaxationHelper(const AtomicRelaxParamsPointers& shared,
                           ElementId                        el_id);

    // Whether atomic relaxation should be applied
    explicit inline CELER_FUNCTION operator bool() const;

    // Space needed for relaxation secondaries
    inline CELER_FUNCTION size_type max_secondaries() const;

    // Space needed for subshell ID stack
    inline CELER_FUNCTION size_type max_vacancies() const;

    // Create the sampling distribution from sampled shell and allocated mem
    inline CELER_FUNCTION AtomicRelaxation
    build_distribution(SubshellId       shell_id,
                       Span<Secondary>  secondaries,
                       Span<SubshellId> vacancies) const;

  private:
    // Shared EADL atomic relaxation data
    const AtomicRelaxParamsPointers& shared_;
    // Index in MaterialParams elements
    ElementId el_id_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "AtomicRelaxationHelper.i.hh"

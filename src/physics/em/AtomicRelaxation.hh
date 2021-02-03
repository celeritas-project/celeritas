//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "random/distributions/IsotropicDistribution.hh"
#include "AtomicRelaxationInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Atomic relaxation.
 *
 * The EADL radiative and non-radiative transition data is used to simulate the
 * emission of fluorescence photons and (optionally) Auger electrons given an
 * initial shell vacancy created by a primary process.
 */
class AtomicRelaxation
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    AtomicRelaxation(const AtomicRelaxParamsPointers& shared,
                     ElementId                        el_id,
                     Span<Secondary>                  secondaries);

    // Simulate atomic relaxation with an initial vacancy in the given shell ID
    template<class Engine>
    inline CELER_FUNCTION size_type operator()(size_type shell_id, Engine& rng);

  private:
    // Shared EADL atomic relaxation data
    const AtomicRelaxParamsPointers& shared_;
    // Index in MaterialParams elements
    ElementId el_id_;
    // Fluorescence photons and Auger electrons
    Span<Secondary> secondaries_;
    // Number of vacancy shells remaining to be processed
    size_type num_vacancies_;
    // Stack of vacancy shell IDs to be processed
    // TODO: How should we allocate storage for these vacancies? Possibly use
    // StackAllocator? If we are only simulating radiative transitions, there
    // is only ever 1 vacancy waiting to be processed. For non-radiative
    // transitions, the upper bound on the maximum number of vacancies in the
    // stack at one time is n, where n is the number of shells containing
    // transition data for a given element (19 for Z = 100). But in practice
    // that won't happen, and we could probably bound it closer to 5 for even
    // the highest Z.
    Array<size_type, 10> vacancies_;
    // Angular distribution of secondaries
    IsotropicDistribution<real_type> sample_direction_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "AtomicRelaxation.i.hh"

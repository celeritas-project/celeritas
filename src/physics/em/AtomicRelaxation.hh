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

    struct result_type
    {
        Span<Secondary> secondaries;
        real_type energy; //! Sum of the energies of the secondaries
    };

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    AtomicRelaxation(const AtomicRelaxParamsPointers& shared,
                     ElementId                        el_id,
                     SubshellId                       shell_id,
                     Span<Secondary>                  secondaries,
                     Span<SubshellId>                 vacancies,
                     size_type                        base_size = 0);

    // Simulate atomic relaxation with an initial vacancy in the given shell ID
    template<class Engine>
    inline CELER_FUNCTION result_type operator()(Engine& rng);

  private:
    // Shared EADL atomic relaxation data
    const AtomicRelaxParamsPointers& shared_;
    // Index in MaterialParams elements
    ElementId el_id_;
    // Shell ID of the initial vacancy
    SubshellId shell_id_;
    // Fluorescence photons and Auger electrons
    Span<Secondary> secondaries_;
    // Stack to store unprocessed subshell vacancies
    Span<SubshellId> vacancies_;
    // The number of secondaries already created by the primary process
    size_type base_size_;
    // Angular distribution of secondaries
    IsotropicDistribution<real_type> sample_direction_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "AtomicRelaxation.i.hh"

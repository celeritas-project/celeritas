//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/em/AtomicRelaxationInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for calculating the maximum possible number of secondaries
 * produced in atomic relaxation for a given element and electron/photon
 * production threshold.
 */
class MaxSecondariesCalculator
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //!@}

  public:
    // Construct with EADL transition data and production thresholds
    MaxSecondariesCalculator(const AtomicRelaxElement& el,
                             MevEnergy                 electron_cut,
                             MevEnergy                 gamma_cut);

    // Calculate the maximum possible number of secondaries produced
    size_type operator()();

  private:
    Span<const AtomicRelaxSubshell>           shells_;
    const real_type                           electron_cut_;
    const real_type                           gamma_cut_;
    std::unordered_map<SubshellId, size_type> visited_;

    // HELPER FUNCTIONS

    size_type calc(SubshellId vacancy_shell, size_type count);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

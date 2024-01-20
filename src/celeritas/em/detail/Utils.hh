//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "../data/AtomicRelaxationData.hh"

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
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Values = HostCRef<AtomicRelaxParamsData>;
    //!@}

  public:
    // Construct with EADL transition data and production thresholds
    MaxSecondariesCalculator(Values const& data,
                             ItemRange<AtomicRelaxSubshell> const& shells,
                             Energy electron_cut,
                             Energy gamma_cut);

    // Calculate the maximum possible number of secondaries produced
    size_type operator()();

  private:
    Values const& data_;
    Span<AtomicRelaxSubshell const> shells_;
    Energy const electron_cut_;
    Energy const gamma_cut_;
    std::unordered_map<SubshellId, size_type> visited_;

    // HELPER FUNCTIONS

    size_type calc(SubshellId vacancy_shell, size_type count);
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for calculating the maximum size the stack of unprocessed
 * subshell vacancies can grow to for a given element while simulating the
 * cascade of photons and electrons in atomic relaxation.
 */
class MaxStackSizeCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Values = HostCRef<AtomicRelaxParamsData>;
    //!@}

  public:
    // Construct with EADL transition data
    MaxStackSizeCalculator(Values const& data,
                           ItemRange<AtomicRelaxSubshell> const& shells);

    // Calculate the maximum size of the stack
    size_type operator()();

  private:
    Values const& data_;
    Span<AtomicRelaxSubshell const> shells_;
    std::unordered_map<SubshellId, size_type> visited_;

    // HELPER FUNCTIONS

    size_type calc(SubshellId vacancy_shell);
};

//---------------------------------------------------------------------------//
// Calculate the maximum possible secondaries produced in atomic relaxation
size_type calc_max_secondaries(MaxSecondariesCalculator::Values const& data,
                               ItemRange<AtomicRelaxSubshell> const& shells,
                               units::MevEnergy electron_cut,
                               units::MevEnergy gamma_cut);

// Calculate the maximum size of the vacancy stack in atomic relaxation
size_type calc_max_stack_size(MaxStackSizeCalculator::Values const& data,
                              ItemRange<AtomicRelaxSubshell> const& shells);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

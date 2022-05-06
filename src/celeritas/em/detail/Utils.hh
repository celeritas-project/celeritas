//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
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
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    using Values
        = AtomicRelaxParamsData<Ownership::const_reference, MemSpace::host>;
    //!@}

  public:
    // Construct with EADL transition data and production thresholds
    MaxSecondariesCalculator(const Values&                         data,
                             const ItemRange<AtomicRelaxSubshell>& shells,
                             MevEnergy electron_cut,
                             MevEnergy gamma_cut);

    // Calculate the maximum possible number of secondaries produced
    size_type operator()();

  private:
    const Values&                             data_;
    Span<const AtomicRelaxSubshell>           shells_;
    const real_type                           electron_cut_;
    const real_type                           gamma_cut_;
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
    //! Type aliases
    using Values
        = AtomicRelaxParamsData<Ownership::const_reference, MemSpace::host>;
    //!@}

  public:
    // Construct with EADL transition data
    MaxStackSizeCalculator(const Values&                         data,
                           const ItemRange<AtomicRelaxSubshell>& shells);

    // Calculate the maximum size of the stack
    size_type operator()();

  private:
    const Values&                             data_;
    Span<const AtomicRelaxSubshell>           shells_;
    std::unordered_map<SubshellId, size_type> visited_;

    // HELPER FUNCTIONS

    size_type calc(SubshellId vacancy_shell);
};

//---------------------------------------------------------------------------//
// Calculate the maximum possible secondaries produced in atomic relaxation
size_type calc_max_secondaries(const MaxSecondariesCalculator::Values& data,
                               const ItemRange<AtomicRelaxSubshell>&   shells,
                               units::MevEnergy electron_cut,
                               units::MevEnergy gamma_cut);

// Calculate the maximum size of the vacancy stack in atomic relaxation
size_type calc_max_stack_size(const MaxStackSizeCalculator::Values& data,
                              const ItemRange<AtomicRelaxSubshell>& shells);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

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
 * produced in atomic relaxation when the initial vacancy is in the given
 * subshell.
 */
class MaxSecondariesCalculator
{
  public:
    // Construct with EADL transition data
    MaxSecondariesCalculator(const AtomicRelaxElement& el);

    // Calculate the maximum possible secondaries produced
    size_type operator()();

  private:
    Span<const AtomicRelaxSubshell>           shells_;
    std::unordered_map<SubshellId, size_type> visited_;

    // HELPER FUNCTIONS

    size_type calc(SubshellId vacancy_shell, size_type count);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <cmath>
#include "base/Algorithms.hh"
#include "base/Range.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with EADL transition data.
 */
MaxSecondariesCalculator::MaxSecondariesCalculator(const AtomicRelaxElement& el)
    : shells_(el.shells)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum possible number of secondaries produced in atomic
 * relaxation.
 */
size_type MaxSecondariesCalculator::operator()()
{
    // No atomic relaxation data for this element
    if (shells_.empty())
        return 0;

    // Find the maximum number of secondaries created, checking over every
    // possible subshell the initial vacancy could be in
    size_type result = 0;
    for (SubshellId::value_type shell_idx : range(shells_.size()))
        result = max(result, this->calc(SubshellId{shell_idx}, 0));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for calculating the maximum possible number of secondaries
 * when the initial vacancy is in the given subshell.
 */
size_type
MaxSecondariesCalculator::calc(SubshellId vacancy_shell, size_type count)
{
    // No transitions for this subshell, so no secondaries produced
    if (!vacancy_shell || vacancy_shell.get() >= shells_.size())
        return 0;

    auto it = visited_.find(vacancy_shell);
    if (it != visited_.end())
        return count + it->second;

    size_type sub_count = 0;
    for (const auto& transition : shells_[vacancy_shell.get()].transitions)
    {
        sub_count = std::max(1 + this->calc(transition.initial_shell, count)
                                 + this->calc(transition.auger_shell, count),
                             sub_count);
    }
    visited_[vacancy_shell] = sub_count;
    return count + sub_count;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

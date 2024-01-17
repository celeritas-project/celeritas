//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/detail/Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <algorithm>
#include <utility>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with EADL transition data and production thresholds.
 */
MaxSecondariesCalculator::MaxSecondariesCalculator(
    Values const& data,
    ItemRange<AtomicRelaxSubshell> const& shells,
    Energy electron_cut,
    Energy gamma_cut)
    : data_(data)
    , shells_(data.shells[shells])
    , electron_cut_(electron_cut)
    , gamma_cut_(gamma_cut)
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
    for (auto shell_id : range(SubshellId(shells_.size())))
        result = max(result, this->calc(shell_id, 0));
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
    for (auto const& transition :
         data_.transitions[shells_[vacancy_shell.get()].transitions])
    {
        // If this is a non-radiative transition with an energy above the
        // electron production threshold, create an electron; if this is a
        // radiative transition with an energy above the gamma production
        // threshold, create a photon; otherwise, no secondaries produced.
        size_type n
            = ((transition.energy >= electron_cut_ && transition.auger_shell)
               || (transition.energy >= gamma_cut_ && !transition.auger_shell))
                  ? 1
                  : 0;

        sub_count = std::max(n + this->calc(transition.initial_shell, count)
                                 + this->calc(transition.auger_shell, count),
                             sub_count);
    }
    visited_[vacancy_shell] = sub_count;
    return count + sub_count;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with EADL transition data.
 */
MaxStackSizeCalculator::MaxStackSizeCalculator(
    Values const& data, ItemRange<AtomicRelaxSubshell> const& shells)
    : data_(data), shells_(data.shells[shells])
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum size of the vacancy stack in atomic relaxation.
 */
size_type MaxStackSizeCalculator::operator()()
{
    // No atomic relaxation data for this element
    if (shells_.empty())
        return 0;

    // Find the maximum possible size of the stack, checking over every
    // subshell the initial vacancy could be in
    size_type result = 0;
    for (auto shell_id : range(SubshellId(shells_.size())))
        result = max(result, this->calc(shell_id));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for calculating the maximum possible size the stack can grow
 * to when the initial vacancy is in the given subshell.
 */
size_type MaxStackSizeCalculator::calc(SubshellId vacancy_shell)
{
    // No transitions for this subshell, so this is the only shell in the stack
    if (vacancy_shell.get() >= shells_.size())
        return 1;

    // Check the table to see if the maximum stack size has already been
    // calculated for this shell
    auto it = visited_.find(vacancy_shell);
    if (it != visited_.end())
        return it->second;

    size_type max_depth = 0;
    for (auto const& transition :
         data_.transitions[shells_[vacancy_shell.get()].transitions])
    {
        // If this is a non-radiative transition two vacancies are created and
        // the stack grows by one; if this is a radiative transition only one
        // vacancy is created and the stack size stays the same
        size_type depth = 0;
        if (transition.auger_shell)
        {
            depth = this->calc(transition.auger_shell) + 1;
        }
        depth = max(depth, this->calc(transition.initial_shell));
        max_depth = max(max_depth, depth);
    }
    visited_[vacancy_shell] = max_depth;
    return max_depth;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum possible number of secondaries produced in atomic
 * relaxation.
 */
size_type calc_max_secondaries(MaxSecondariesCalculator::Values const& data,
                               ItemRange<AtomicRelaxSubshell> const& shells,
                               units::MevEnergy electron_cut,
                               units::MevEnergy gamma_cut)
{
    return MaxSecondariesCalculator(data, shells, electron_cut, gamma_cut)();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum size the stack of subshell vacancies can grow to in
 * atomic relaxation.
 */
size_type calc_max_stack_size(MaxSecondariesCalculator::Values const& data,
                              ItemRange<AtomicRelaxSubshell> const& shells)
{
    return MaxStackSizeCalculator(data, shells)();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

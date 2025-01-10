//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/DepthCalculator.cc
//---------------------------------------------------------------------------//
#include "DepthCalculator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to all universe inputs.
 */
DepthCalculator::DepthCalculator(VecVarUniv const& inp)
    : visit_univ_{inp}, num_univ_{inp.size()}
{
    CELER_EXPECT(num_univ_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the depth of the global unit.
 */
size_type DepthCalculator::operator()()
{
    return this->visit_univ_(*this, 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the depth of a unit.
 */
size_type DepthCalculator::operator()(UnitInput const& u)
{
    // Calculate the depth of the deepest daughter
    size_type max_daughter{0};
    for (auto&& [vol, daughter] : u.daughter_map)
    {
        max_daughter = std::max(max_daughter, (*this)(daughter.universe_id));
    }

    // Add one for the current universe
    return max_daughter + 1;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the depth of a rect array.
 */
size_type DepthCalculator::operator()(RectArrayInput const& u)
{
    // Calculate the depth of the deepest daughter
    size_type max_daughter{0};
    for (auto&& daughter : u.daughters)
    {
        max_daughter = std::max(max_daughter, (*this)(daughter.universe_id));
    }

    // Add one for the current universe
    return max_daughter + 1;
}

//---------------------------------------------------------------------------//
/*!
 * Check cache or calculate.
 */
size_type DepthCalculator::operator()(UniverseId univ_id)
{
    CELER_EXPECT(univ_id < num_univ_);
    // Check for cached value
    auto&& [iter, inserted] = depths_.insert({univ_id, {}});
    if (inserted)
    {
        // Visit and save value
        iter->second = visit_univ_(*this, univ_id.unchecked_get());
    }

    // Return cached value
    CELER_ENSURE(iter->second > 0);
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/DepthCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "corecel/cont/VariantUtils.hh"

#include "OrangeInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum number of levels deep in a geometry.
 */
class DepthCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using VecVarUniv = std::vector<VariantUniverseInput>;
    //!@}

  public:
    // Construct with a reference to all universe inputs
    explicit DepthCalculator(VecVarUniv const&);

    //! Calculate the depth of the global unit
    size_type operator()() { return this->visit_univ_(*this, 0); }

    // Calculate the depth of a unit
    size_type operator()(UnitInput const& u);

    // Calculate the depth of a rect array
    size_type operator()(RectArrayInput const& u);

  private:
    ContainerVisitor<VecVarUniv const&> visit_univ_;
    std::size_t num_univ_;
    std::unordered_map<UniverseId, size_type> depths_;

    // Check cache or calculate
    size_type operator()(UniverseId uid);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

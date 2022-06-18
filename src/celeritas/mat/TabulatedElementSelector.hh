//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/TabulatedElementSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Make a weighted random selection of an element.
 *
 * This selects an elemental component (atom) of a material based on the
 * precalculated cross section CDF tables of the elements in the material.
 * Unlike \c ElementSelector which calculates the microscopic cross sections on
 * the fly, this interpolates the values using tabulated CDF grids.
 */
class TabulatedElementSelector
{
  public:
    //!@{
    //! Type aliases
    using Energy = Quantity<XsGridData::EnergyUnits>;
    using GridValues
        = Collection<ValueGrid, Ownership::const_reference, MemSpace::native>;
    using GridIdValues
        = Collection<ValueGridId, Ownership::const_reference, MemSpace::native>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with xs CDF data for a particular model and material
    inline CELER_FUNCTION TabulatedElementSelector(const ValueTable&   table,
                                                   const GridValues&   grids,
                                                   const GridIdValues& ids,
                                                   const Values&       reals,
                                                   Energy              energy);

    // Sample with the given RNG
    template<class Engine>
    inline CELER_FUNCTION ElementComponentId operator()(Engine& rng) const;

  private:
    const ValueTable&   table_;
    const GridValues&   grids_;
    const GridIdValues& ids_;
    const Values&       reals_;
    const Energy        energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with xs CDF data for a particular model and material.
 */
CELER_FUNCTION
TabulatedElementSelector::TabulatedElementSelector(const ValueTable&   table,
                                                   const GridValues&   grids,
                                                   const GridIdValues& ids,
                                                   const Values&       reals,
                                                   Energy              energy)
    : table_(table), grids_(grids), ids_(ids), reals_(reals), energy_(energy)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the element with the given RNG.
 */
template<class Engine>
CELER_FUNCTION ElementComponentId
TabulatedElementSelector::operator()(Engine& rng) const
{
    if (!table_)
    {
        // Material only has a single element
        return ElementComponentId{0};
    }

    size_type i = 0;
    real_type u = generate_canonical(rng);
    for (; i < table_.grids.size(); ++i)
    {
        ValueGridId grid_id = ids_[table_.grids[i]];
        CELER_ASSERT(grid_id < grids_.size());

        XsCalculator calc_xs(grids_[grid_id], reals_);
        if (calc_xs(energy_) > u)
            break;
    }
    return ElementComponentId{i};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

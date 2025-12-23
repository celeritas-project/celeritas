//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/detail/SBTableInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"
#include "celeritas/grid/TwodGridBuilder.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct Seltzer-Berger differential cross section data from imported data.
 */
class SBTableInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<SeltzerBergerTableData>;
    using GridInput = inp::TwodGrid;
    //!@}

  public:
    // Construct with pointer to host data
    explicit inline SBTableInserter(Data* data);

    // Construct differential cross section table for a single element
    inline void operator()(GridInput const& inp);

  private:
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;

    TwodGridBuilder build_grid_;
    CollectionBuilder<size_type> argmax_;
    CollectionBuilder<SBElementTableData, MemSpace::host, ElementId> elements_;
    Values const& reals_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with data.
 */
SBTableInserter::SBTableInserter(Data* data)
    : build_grid_{&data->reals}
    , argmax_{&data->sizes}
    , elements_{&data->elements}
    , reals_(data->reals)
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
/*!
 * Construct differential cross section tables for a single element.
 *
 * Here, x is the log of scaled incident energy (E / MeV), y is the scaled
 * exiting energy (E_gamma / E_inc), and values are the cross sections.
 */
void SBTableInserter::operator()(GridInput const& inp)
{
    CELER_EXPECT(inp);

    SBElementTableData table;
    table.grid = build_grid_(inp);

    size_type const num_x = inp.x.size();
    size_type const num_y = inp.y.size();

    // Find the location of the highest cross section at each incident E
    std::vector<size_type> argmax(num_x);
    for (size_type i : range(num_x))
    {
        // Get the xs data for the given incident energy coordinate
        real_type const* iter = &reals_[table.grid.at(i, 0)];

        // Search for the highest cross section value
        size_type max_el = std::max_element(iter, iter + num_y) - iter;
        CELER_ASSERT(max_el < num_y);
        // Save it!
        argmax[i] = max_el;
    }
    table.argmax = argmax_.insert_back(argmax.begin(), argmax.end());

    // Add the table
    elements_.push_back(table);

    CELER_ENSURE(table.grid.x.size() == num_x);
    CELER_ENSURE(table.grid.y.size() == num_y);
    CELER_ENSURE(table.argmax.size() == num_x);
    CELER_ENSURE(table.grid);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

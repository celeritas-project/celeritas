//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/detail/LivermoreXsInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/io/ImportLivermorePE.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct Livermore Photoelectric cross section data from imported data.
 */
class LivermoreXsInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<LivermorePEXsData>;
    //!@}

  public:
    // Construct with pointer to host data
    explicit inline LivermoreXsInserter(Data* data);

    // Construct cross section data for a single element
    inline void operator()(ImportLivermorePE const& inp);

  private:
    NonuniformGridBuilder build_grid_;

    CollectionBuilder<LivermoreSubshell> shells_;
    CollectionBuilder<LivermoreElement, MemSpace::host, ElementId> elements_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with data.
 */
LivermoreXsInserter::LivermoreXsInserter(Data* data)
    : build_grid_{&data->reals}
    , shells_{&data->shells}
    , elements_{&data->elements}
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
/*!
 * Construct cross section data for a single element.
 */
void LivermoreXsInserter::operator()(ImportLivermorePE const& inp)
{
    CELER_EXPECT(!inp.shells.empty());
    if constexpr (CELERITAS_DEBUG)
    {
        CELER_EXPECT(inp.thresh_lo <= inp.thresh_hi);
        for (auto const& shell : inp.shells)
        {
            CELER_EXPECT(shell.param_lo.size() == 6);
            CELER_EXPECT(shell.param_hi.size() == 6);
            CELER_EXPECT(shell.binding_energy <= inp.thresh_lo);
        }
    }
    using units::MevEnergy;

    LivermoreElement el;

    // Add tabulated total cross sections
    if (inp.xs_lo)
    {
        // Z < 3 have no low-energy cross sections
        el.xs_lo = build_grid_(inp.xs_lo);
    }
    el.xs_hi = build_grid_(inp.xs_hi);

    // Add energy thresholds for using low and high xs parameterization
    el.thresh_lo = MevEnergy(inp.thresh_lo);
    el.thresh_hi = MevEnergy(inp.thresh_hi);

    // Allocate subshell data
    std::vector<LivermoreSubshell> shells(inp.shells.size());

    // Add subshell data
    for (auto i : range(inp.shells.size()))
    {
        // Ionization energy
        shells[i].binding_energy = MevEnergy(inp.shells[i].binding_energy);

        // Tabulated subshell cross section
        shells[i].xs = build_grid_(make_span(inp.shells[i].energy),
                                   make_span(inp.shells[i].xs));

        // Subshell cross section fit parameters
        std::copy(inp.shells[i].param_lo.begin(),
                  inp.shells[i].param_lo.end(),
                  shells[i].param[0].begin());
        std::copy(inp.shells[i].param_hi.begin(),
                  inp.shells[i].param_hi.end(),
                  shells[i].param[1].begin());

        CELER_ASSERT(shells[i]);
    }
    el.shells = shells_.insert_back(shells.begin(), shells.end());

    // Add the elemental data
    CELER_ASSERT(el);
    elements_.push_back(el);

    CELER_ENSURE(el.shells.size() == inp.shells.size());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

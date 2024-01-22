//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/XsGridParams.cc
//---------------------------------------------------------------------------//
#include "XsGridParams.hh"

#include <algorithm>
#include <cmath>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/grid/XsGridData.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
namespace
{
bool is_same_log_grid(UniformGridData const& grid,
                      std::vector<real_type> const& energy,
                      real_type tol)
{
    UniformGrid log_energy(grid);
    if (log_energy.size() != energy.size())
    {
        return false;
    }

    auto r = range(energy.size());
    SoftEqual const soft_eq{tol};
    return std::all_of(r.begin(), r.end(), [&](auto i) {
        return soft_eq(std::log(energy[i]), log_energy[i]);
    });
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with input data.
 */
XsGridParams::XsGridParams(Input const& input)
{
    CELER_EXPECT(input.energy.size() >= 2);
    CELER_EXPECT(input.energy.front() > 0);
    CELER_EXPECT(std::is_sorted(input.energy.begin(), input.energy.end()));
    CELER_EXPECT(input.xs.size() == input.energy.size());
    CELER_EXPECT(std::all_of(
        input.xs.begin(), input.xs.end(), [](real_type v) { return v >= 0; }));

    TableData<Ownership::value, MemSpace::host> host_data;

    // Construct cross section
    XsGridData& host_xs = host_data.xs;
    host_xs.log_energy
        = UniformGridData::from_bounds(std::log(input.energy.front()),
                                       std::log(input.energy.back()),
                                       input.energy.size());
    CELER_ASSERT(is_same_log_grid(host_xs.log_energy, input.energy, 1e-3));
    host_xs.prime_index = std::find(input.energy.begin(),
                                    input.energy.end(),
                                    input.prime_energy)
                          - input.energy.begin();
    CELER_EXPECT(host_xs.prime_index != input.energy.size());
    host_xs.value = make_builder(&host_data.reals)
                        .insert_back(input.xs.begin(), input.xs.end());

    data_ = CollectionMirror<TableData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

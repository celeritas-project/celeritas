//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CherenkovParams.cc
//---------------------------------------------------------------------------//
#include "CherenkovParams.hh"

#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"
#include "celeritas/grid/GenericGridInserter.hh"

#include "MaterialParams.hh"
#include "MaterialView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
CherenkovParams::CherenkovParams(MaterialParams const& mats)
{
    HostVal<CherenkovData> data;
    GenericGridInserter insert_angle_integral(&data.reals,
                                              &data.angle_integral);
    for (auto mat_id : range(OpticalMaterialId(mats.num_materials())))
    {
        GenericCalculator refractive_index
            = MaterialView{mats.host_ref(), mat_id}
                  .make_refractive_index_calculator();
        Span<real_type const> energy = refractive_index.grid().values();

        // Calculate the Cherenkov angle integral
        std::vector<real_type> integral(energy.size());
        for (size_type i = 1; i < energy.size(); ++i)
        {
            // TODO: use trapezoidal integrator helper class
            integral[i] = integral[i - 1]
                          + real_type(0.5) * (energy[i] - energy[i - 1])
                                * (1 / ipow<2>(refractive_index[i - 1])
                                   + 1 / ipow<2>(refractive_index[i]));
        }
        insert_angle_integral(energy, make_span(integral));
    }
    CELER_ASSERT(data.angle_integral.size() == mats.num_materials());
    data_ = CollectionMirror<CherenkovData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

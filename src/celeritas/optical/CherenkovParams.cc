//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/math/CdfUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"

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
    SegmentIntegrator integrate_rindex{TrapezoidSegmentIntegrator{}};

    HostVal<CherenkovData> data;
    NonuniformGridInserter insert_angle_integral(&data.reals,
                                                 &data.angle_integral);
    for (auto mat_id : range(OpticalMaterialId(mats.num_materials())))
    {
        NonuniformGridCalculator refractive_index
            = MaterialView{mats.host_ref(), mat_id}
                  .make_refractive_index_calculator();
        Span<real_type const> energy = refractive_index.grid().values();

        // Calculate 1/n^2 on all grid points
        std::vector<real_type> ri_inv_sq(energy.size());
        for (auto i : range(ri_inv_sq.size()))
        {
            ri_inv_sq[i] = 1 / ipow<2>(refractive_index[i]);
        }

        // Integrate
        std::vector<real_type> integral(energy.size());
        integrate_rindex(energy,
                         Span<real_type const>(make_span(ri_inv_sq)),
                         make_span(integral));
        insert_angle_integral(energy, make_span(integral));
    }
    CELER_ASSERT(data.angle_integral.size() == mats.num_materials());
    data_ = CollectionMirror<CherenkovData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

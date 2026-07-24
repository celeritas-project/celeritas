//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/CherenkovParams.cc
//---------------------------------------------------------------------------//
#include "CherenkovParams.hh"

#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/PdfUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"

#include "../MaterialParams.hh"
#include "../MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
CherenkovParams::CherenkovParams(optical::MaterialParams const& mats)
{
    SegmentIntegrator integrate_rindex{TrapezoidSegmentIntegrator{}};

    HostVal<CherenkovData> data;
    NonuniformGridInserter insert_angle_integral(&data.reals,
                                                 &data.angle_integral);
    for (auto mat_id : range(OptMatId(mats.num_materials())))
    {
        NonuniformGridCalculator refractive_index
            = optical::MaterialView{mats.host_ref(), mat_id}
                  .make_refractive_index_calculator();
        auto energy = refractive_index.grid().values();

        // The Cherenkov photon sampler inverts the refractive index to find
        // the emission energy range, which requires monotonic data. The
        // material params accept anomalous dispersion (needed only for
        // transport), so enforce the stricter requirement here.
        {
            std::vector<double> rindex_values(energy.size());
            for (auto i : range(rindex_values.size()))
            {
                rindex_values[i] = refractive_index[i];
            }
            CELER_VALIDATE(
                is_monotonic_nondecreasing(make_span(rindex_values)),
                << "refractive index for optical material " << mat_id.get()
                << " is not monotonically increasing with photon energy, "
                   "which Cherenkov photon generation requires");
        }

        inp::Grid grid;
        grid.x = {energy.begin(), energy.end()};

        // Calculate 1/n^2 on all grid points
        std::vector<double> ri_inv_sq(grid.x.size());
        for (auto i : range(ri_inv_sq.size()))
        {
            ri_inv_sq[i] = 1 / ipow<2>(refractive_index[i]);
        }

        // Integrate
        grid.y.resize(grid.x.size());
        integrate_rindex(Span<double const>(make_span(grid.x)),
                         Span<double const>(make_span(ri_inv_sq)),
                         make_span(grid.y));
        insert_angle_integral(grid);
    }
    CELER_ASSERT(data.angle_integral.size() == mats.num_materials());
    data_ = ParamsDataStore<CherenkovData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

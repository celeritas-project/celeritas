//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

#include "GridReflectivityData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample user-defined reflectivity and transmittance grids to determine if the
 * track is transmitted, absorbed, or undergoes usual physics interactions.
 */
struct GridReflectivityExecutor
{
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

    NativeCRef<GridReflectivityData> data;

    //! Apply grid reflectivity executor
    CELER_FUNCTION ReflectivityAction operator()(CoreTrackView const& track) const
    {
        auto s_phys = track.surface_physics();
        auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::reflectivity)
                                .internal_surface_id();
        CELER_ASSERT(sub_model_id < data.reflectivity.size());

        auto const& grid = data.reflectivity[sub_model_id];
        CELER_ASSERT(grid);

        NonuniformGridCalculator calc_reflectivity{grid, data.reals};
        real_type reflectivity
            = calc_reflectivity(value_as<Energy>(track.particle().energy()));

        CELER_ENSURE(0 <= reflectivity && reflectivity <= 1);

        auto rng = track.rng();

        return BernoulliDistribution{reflectivity}(rng)
                   ? ReflectivityAction::interact
                   : ReflectivityAction::absorb;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

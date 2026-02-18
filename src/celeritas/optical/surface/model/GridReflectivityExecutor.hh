//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GridReflectivityExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/Selector.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

#include "GridReflectivityData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class GridReflectivitySampler
{
  public:
    //!@{
    //! \name Type aliases
    using DataRef = NativeCRef<GridReflectivityData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    explicit inline CELER_FUNCTION
    GridReflectivitySampler(DataRef const&, SubModelId, Energy);

    inline CELER_FUNCTION real_type operator()(ReflectivityAction) const;

  private:
    DataRef const& data_;
    SubModelId surface_;
    Energy energy_;
};

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
    inline CELER_FUNCTION ReflectivityAction
    operator()(CoreTrackView const& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION
GridReflectivitySampler::GridReflectivitySampler(DataRef const& data,
                                                 SubModelId surface,
                                                 Energy energy)
    : data_(data), surface_(surface), energy_(energy)
{
}

CELER_FUNCTION real_type
GridReflectivitySampler::operator()(ReflectivityAction action) const
{
    CELER_EXPECT(surface_ < data_.reflectivity[action].size());
    auto grid = data_.reflectivity[action][surface_];

    CELER_ASSERT(grid);

    NonuniformGridCalculator calc_grid{grid, data_.reals};
    real_type result = calc_grid(value_as<Energy>(energy_));

    CELER_ENSURE(0 <= result && result <= 1);

    return result;
}

CELER_FUNCTION ReflectivityAction
GridReflectivityExecutor::operator()(CoreTrackView const& track) const
{
    auto s_phys = track.surface_physics();
    auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::reflectivity)
                            .internal_surface_id();

    auto rng = track.rng();

    // Sample action based on reflectivity and transmittance grids
    auto action = celeritas::make_unnormalized_selector(
        GridReflectivitySampler{data, sub_model_id, track.particle().energy()},
        ReflectivityAction::size_,
        real_type{1})(rng);

    if (action == ReflectivityAction::absorb)
    {
        auto e_grid_id = data.efficiency_ids[sub_model_id];
        if (e_grid_id < data.efficiency.size())
        {
            // If absorbed and has efficiency grid, sample efficiency
            auto const& e_grid = data.efficiency[e_grid_id];

            CELER_ASSERT(e_grid);

            real_type efficiency = NonuniformGridCalculator{e_grid, data.reals}(
                value_as<Energy>(track.particle().energy()));

            if (BernoulliDistribution{efficiency}(rng))
            {
                // Pass efficiency selection; transmit instead of absorb
                action = ReflectivityAction::transmit;
            }
        }
    }

    return action;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

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
 * Calculate the probability of a photon undergoing the specified reflectivity
 * action for a given grid.
 */
class GridReflectivityCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using DataRef = NativeCRef<GridReflectivityData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from data, surface, and energy
    explicit inline CELER_FUNCTION
    GridReflectivityCalculator(DataRef const&, SubModelId, Energy);

    // Calculate the probability for the specified reflectivity action
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
 *
 * If the track is absorbed and a non-zero efficiency grid is present, then it
 * is also sampled. If it passes the efficiency threshold then it is instead
 * set to transmit to the next sub-surface.
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
/*!
 * Construct from data, surface, and energy.
 */
CELER_FUNCTION
GridReflectivityCalculator::GridReflectivityCalculator(DataRef const& data,
                                                       SubModelId surface,
                                                       Energy energy)
    : data_(data), surface_(surface), energy_(energy)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the probability for the specified reflectivity action.
 */
CELER_FUNCTION real_type
GridReflectivityCalculator::operator()(ReflectivityAction action) const
{
    CELER_EXPECT(surface_ < data_.reflectivity[action].size());
    auto grid = data_.reflectivity[action][surface_];

    CELER_ASSERT(grid);

    NonuniformGridCalculator calc_grid{grid, data_.reals};
    real_type result = calc_grid(value_as<Energy>(energy_));

    // Values are probabilities and should be in [0,1]
    CELER_ENSURE(0 <= result && result <= 1);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Apply the executor to a track.
 */
CELER_FUNCTION ReflectivityAction
GridReflectivityExecutor::operator()(CoreTrackView const& track) const
{
    auto s_phys = track.surface_physics();
    auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::reflectivity)
                            .internal_surface_id();

    auto rng = track.rng();

    // Sample action based on reflectivity and transmittance grids
    auto action = celeritas::make_unnormalized_selector(
        GridReflectivityCalculator{
            data, sub_model_id, track.particle().energy()},
        ReflectivityAction::size_,
        real_type{1})(rng);

    if (action == ReflectivityAction::absorb)
    {
        if (auto e_grid_id = data.efficiency_ids[sub_model_id])
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

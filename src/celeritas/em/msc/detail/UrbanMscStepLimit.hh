//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/UrbanMscStepLimit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/PolyEvaluator.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/distribution/NormalDistribution.hh"

#include "MscStepToGeo.hh"
#include "UrbanMscHelper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample a step limit for the Urban MSC model.
 *
 * This distribution is to be used for tracks that have non-negligible steps
 * and are near the boundary. Otherwise, no displacement or step limiting is
 * needed.
 *
 * \note This code performs the same method as in ComputeTruePathLengthLimit
 * of G4UrbanMscModel, as documented in section 8.1.6 of the Geant4 10.7
 * Physics Reference Manual or CERN-OPEN-2006-077 by L. Urban.
 */
class UrbanMscStepLimit
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using UrbanMscRef = NativeCRef<UrbanMscData>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscStepLimit(UrbanMscRef const& shared,
                                            UrbanMscHelper const& helper,
                                            Energy inc_energy,
                                            PhysicsTrackView* physics,
                                            MaterialId matid,
                                            bool on_boundary,
                                            real_type safety,
                                            real_type phys_step);

    // Apply the step limitation algorithm for the e-/e+ MSC with the RNG
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant data
    UrbanMscRef const& shared_;
    // Urban MSC helper class
    UrbanMscHelper const& helper_;

    // Physical step limitation up to this point
    real_type max_step_{};
    // Cached approximation for the minimum step length
    real_type limit_min_{};
    // Limit based on the range and safety
    real_type limit_{};

    //// HELPER FUNCTIONS ////

    // Calculate the minimum of the true path length limit
    inline CELER_FUNCTION real_type calc_limit_min(UrbanMscMaterialData const&,
                                                   Energy const) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscStepLimit::UrbanMscStepLimit(UrbanMscRef const& shared,
                                     UrbanMscHelper const& helper,
                                     Energy inc_energy,
                                     PhysicsTrackView* physics,
                                     MaterialId matid,
                                     bool on_boundary,
                                     real_type safety,
                                     real_type phys_step)
    : shared_(shared), helper_(helper), max_step_(phys_step)
{
    CELER_EXPECT(safety >= 0);
    CELER_EXPECT(safety < helper_.max_step());
    CELER_EXPECT(max_step_ > shared_.params.limit_min_fix());
    CELER_EXPECT(max_step_ <= physics->dedx_range());

    real_type const range = physics->dedx_range();
    auto const& msc_range = physics->msc_range();

    if (!msc_range || on_boundary)
    {
        MscRange new_range;
        // Initialize MSC range cache on the first step in a volume
        // TODO for hadrons/muons: this value is hard-coded for electrons
        new_range.range_fact = physics->scalars().range_fact;
        // XXX the 1 MFP limitation is applied to the *geo* step, not the true
        // step, so this isn't quite right (See UrbanMsc.hh)
        new_range.range_init = max<real_type>(range, helper_.msc_mfp());
        if (helper_.msc_mfp() > physics->scalars().lambda_limit)
        {
            new_range.range_fact *= (real_type(0.75)
                                     + real_type(0.25) * helper_.msc_mfp()
                                           / physics->scalars().lambda_limit);
        }
        new_range.limit_min
            = this->calc_limit_min(shared_.material_data[matid], inc_energy);

        // Store persistent range properties within this tracking volume
        physics->msc_range(new_range);
        // Range is a reference so should be updated
        CELER_ASSERT(msc_range);
    }
    limit_min_ = msc_range.limit_min;

    limit_ = range;
    if (safety < range)
    {
        limit_ = max<real_type>(msc_range.range_fact * msc_range.range_init,
                                physics->scalars().safety_fact * safety);
    }
    limit_ = max<real_type>(limit_, limit_min_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the true path length using the Urban multiple scattering model.
 */
template<class Engine>
CELER_FUNCTION auto UrbanMscStepLimit::operator()(Engine& rng) -> real_type
{
    if (max_step_ <= max(limit_, limit_min_))
    {
        // Skip sampling if the physics step is limiting
        return max_step_;
    }
    if (limit_ <= limit_min_)
    {
        // Skip sampling below the minimum step limit
        return limit_min_;
    }

    // Randomize the limit if this step should be determined by msc
    NormalDistribution<real_type> sample_gauss(
        limit_, real_type(0.1) * (limit_ - limit_min_));
    real_type sampled_limit = sample_gauss(rng);

    // Keep sampled limit between the minimum value and maximum step
    return clamp(sampled_limit, limit_min_, max_step_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the minimum of the true path length limit.
 */
CELER_FUNCTION real_type UrbanMscStepLimit::calc_limit_min(
    UrbanMscMaterialData const& msc, Energy const inc_energy) const
{
    using PolyQuad = PolyEvaluator<real_type, 2>;

    // Calculate minimum step
    PolyQuad calc_min_mfp(2, msc.stepmin_coeff[0], msc.stepmin_coeff[1]);
    real_type xm = helper_.msc_mfp() / calc_min_mfp(inc_energy.value());

    // Scale based on particle type and effective atomic number
    xm *= helper_.scaled_zeff();

    if (inc_energy < shared_.params.min_scaling_energy())
    {
        // Energy is below a pre-defined limit
        xm *= (real_type(0.5)
               + real_type(0.5) * value_as<Energy>(inc_energy)
                     / value_as<Energy>(shared_.params.min_scaling_energy()));
    }

    return max<real_type>(xm, shared_.params.limit_min_fix());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

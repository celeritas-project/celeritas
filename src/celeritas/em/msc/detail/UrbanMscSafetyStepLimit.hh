//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/UrbanMscSafetyStepLimit.hh
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

#include "UrbanMscHelper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample a step limit for the Urban MSC model using the "safety" algorithm.
 *
 * This distribution is to be used for tracks that have non-negligible steps
 * and are near the boundary. Otherwise, no displacement or step limiting is
 * needed.
 *
 * \note This code performs the same method as in ComputeTruePathLengthLimit
 * of G4UrbanMscModel, as documented in section 8.1.6 of the Geant4 10.7
 * Physics Reference Manual or CERN-OPEN-2006-077 by L. Urban.
 */
class UrbanMscSafetyStepLimit
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using UrbanMscRef = NativeCRef<UrbanMscData>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    UrbanMscSafetyStepLimit(UrbanMscRef const& shared,
                            UrbanMscHelper const& helper,
                            ParticleTrackView const& particle,
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

    //// COMMON PROPERTIES ////

    //! Minimum range for an empirical step-function approach
    static CELER_CONSTEXPR_FUNCTION real_type min_range()
    {
        return 1e-3 * units::centimeter;
    }

    //! Maximum step over the range
    static CELER_CONSTEXPR_FUNCTION real_type max_step_over_range()
    {
        return 0.35;
    }

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
UrbanMscSafetyStepLimit::UrbanMscSafetyStepLimit(
    UrbanMscRef const& shared,
    UrbanMscHelper const& helper,
    ParticleTrackView const& particle,
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

    bool use_safety_plus = physics->particle_scalars().step_limit_algorithm
                           == MscStepLimitAlgorithm::safety_plus;
    real_type const range = physics->dedx_range();
    auto const& msc_range = physics->msc_range();

    if (!msc_range || on_boundary)
    {
        MscRange new_range;
        // Initialize MSC range cache on the first step in a volume
        new_range.range_factor = physics->particle_scalars().range_factor;
        new_range.range_init = range;

        // XXX the 1 MFP limitation is applied to the *geo* step, not the true
        // step, so this isn't quite right (See UrbanMsc.hh)
        if (!particle.is_heavy())
        {
            real_type mfp = helper.msc_mfp();
            if (!use_safety_plus && mfp > range)
            {
                new_range.range_init = mfp;
            }
            if (mfp > physics->scalars().lambda_limit)
            {
                real_type c = use_safety_plus ? 0.84 : 0.75;
                new_range.range_factor
                    *= c + (1 - c) * mfp / physics->scalars().lambda_limit;
            }
        }
        new_range.limit_min = this->calc_limit_min(
            shared_.material_data[matid], particle.energy());

        // Store persistent range properties within this tracking volume
        physics->msc_range(new_range);
        // Range is a reference so should be updated
        CELER_ASSERT(msc_range);
    }
    limit_min_ = msc_range.limit_min;

    limit_ = range;
    if (safety < range)
    {
        limit_ = max<real_type>(msc_range.range_factor * msc_range.range_init,
                                physics->scalars().safety_factor * safety);
    }
    limit_ = max<real_type>(limit_, limit_min_);

    if (use_safety_plus)
    {
        real_type rho = UrbanMscSafetyStepLimit::min_range();
        if (range > rho)
        {
            // Calculate the scaled step range \f$ s = \alpha r + \rho (1 -
            // \alpha) (2 - \frac{\rho}{r}) \f$, where \f$ \alpha \f$ is the
            // maximum step over the range and \f$ \rho \f$ is the minimum
            // range
            real_type alpha = UrbanMscSafetyStepLimit::max_step_over_range();
            real_type limit_step = alpha * range
                                   + rho * (1 - alpha) * (2 - rho / range);
            max_step_ = min(max_step_, limit_step);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample the true path length using the Urban multiple scattering model.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscSafetyStepLimit::operator()(Engine& rng)
{
    if (max_step_ <= limit_)
    {
        // Skip sampling if the physics step is limiting
        return max_step_;
    }
    if (limit_ == limit_min_)
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
 *
 * See \c G4UrbanMscModel::ComputeTlimitmin .
 */
CELER_FUNCTION real_type UrbanMscSafetyStepLimit::calc_limit_min(
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

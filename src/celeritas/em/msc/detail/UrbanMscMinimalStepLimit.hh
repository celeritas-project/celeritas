//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/UrbanMscMinimalStepLimit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

#include "UrbanMscHelper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample a step limit for the Urban MSC model using the "minimal" algorithm.
 *
 * \note This code performs the same method as in ComputeTruePathLengthLimit
 * of G4UrbanMscModel, as documented in section 8.1.6 of \cite{g4prm}
 * or \citet{urban-msc-2006, https://cds.cern.ch/record/1004190/}.
 *
 * \todo Here and \c UrbanMscSafetyStepLimit should simply calculate \c limit
 * and \c limit_min; the caller should skip sampling if not MSC limited (or
 * below min sampling step), and another helper (documented the hardcoded 0.1
 * sigma width) does the gaussian sampling.
 */
class UrbanMscMinimalStepLimit
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    UrbanMscMinimalStepLimit(NativeCRef<UrbanMscData> const& shared,
                             UrbanMscHelper const& helper,
                             PhysicsTrackView* physics,
                             bool on_boundary,
                             real_type phys_step);

    // Apply the step limitation algorithm for e-/e+ MSC
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    //// DATA ////

    // Physical step limitation up to this point
    real_type max_step_{};
    // Cached approximation for the minimum step length
    real_type limit_min_{};
    // Step limit based on the range
    real_type limit_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscMinimalStepLimit::UrbanMscMinimalStepLimit(
    NativeCRef<UrbanMscData> const& shared,
    UrbanMscHelper const& helper,
    PhysicsTrackView* physics,
    bool on_boundary,
    real_type phys_step)
    : max_step_(phys_step)
{
    CELER_EXPECT(max_step_ > shared.params.min_step);
    CELER_EXPECT(max_step_ <= physics->dedx_range());

    auto const& msc_range = physics->msc_range();

    if (!msc_range)
    {
        // Store initial range properties if this is the track's first step
        MscRange new_range;
        new_range.range_init = numeric_limits<real_type>::infinity();
        new_range.range_factor = physics->particle_scalars().range_factor;
        new_range.limit_min = 10 * shared.params.min_step;
        physics->msc_range(new_range);
        CELER_ASSERT(msc_range);
    }
    limit_min_ = msc_range.limit_min;

    if (on_boundary)
    {
        // Update the MSC range for the new volume
        MscRange new_range = msc_range;
        new_range.range_init = msc_range.range_factor
                               * max(physics->dedx_range(), helper.msc_mfp());
        new_range.range_init = max(new_range.range_init, limit_min_);
        physics->msc_range(new_range);
        CELER_ASSERT(msc_range);
    }
    limit_ = msc_range.range_init;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the true path length using the Urban multiple scattering model.
 *
 * \todo This is identical to UrbanMscSafetyStepLimit::operator()
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscMinimalStepLimit::operator()(Engine& rng)
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

    // Randomize the limit if this step should be determined by MSC
    NormalDistribution<real_type> sample_gauss(
        limit_, real_type(0.1) * (limit_ - limit_min_));
    real_type sampled_limit = sample_gauss(rng);

    // Keep sampled limit between the minimum value and maximum step
    return clamp(sampled_limit, limit_min_, max_step_);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

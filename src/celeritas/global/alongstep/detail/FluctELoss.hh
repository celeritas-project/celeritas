//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/FluctELoss.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/distribution/EnergyLossHelper.hh"
#include "celeritas/em/distribution/EnergyLossTraits.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss (with fluctuations) to a track.
 */
class FluctELoss
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = EnergyLossHelper::Energy;
    using ParamsRef = NativeCRef<FluctuationData>;
    //!@}

  public:
    // Construct with flucutation data
    inline explicit CELER_FUNCTION FluctELoss(ParamsRef const& params);

    // Whether energy loss can be used for this track
    inline CELER_FUNCTION bool is_applicable(CoreTrackView const&) const;

    // Apply to the track
    inline CELER_FUNCTION Energy calc_eloss(CoreTrackView const& track,
                                            real_type step);

    //! Indicate that we can lose all energy before hitting the dE/dx range
    static CELER_CONSTEXPR_FUNCTION bool imprecise_range() { return true; }

  private:
    //// DATA ////

    //! Reference to fluctuation data
    ParamsRef const& fluct_params_;

    //// HELPER FUNCTIONS ////

    template<EnergyLossFluctuationModel M>
    inline CELER_FUNCTION Energy sample_energy_loss(
        EnergyLossHelper const& helper, Energy max_loss, RngEngine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to fluctuation data.
 */
CELER_FUNCTION FluctELoss::FluctELoss(ParamsRef const& params)
    : fluct_params_{params}
{
}

//---------------------------------------------------------------------------//
/*!
 * Whether energy loss is used for this track.
 */
CELER_FUNCTION bool FluctELoss::is_applicable(CoreTrackView const& track) const
{
    // Energy loss grid ID will be 'false' if inapplicable
    auto ppid = track.make_physics_view().eloss_ppid();
    return static_cast<bool>(ppid);
}

//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION auto
FluctELoss::calc_eloss(CoreTrackView const& track, real_type step) -> Energy
{
    CELER_EXPECT(step > 0);

    auto particle = track.make_particle_view();

    // Calculate mean energy loss
    auto phys = track.make_physics_view();
    auto eloss = calc_mean_energy_loss(particle, phys, step);

    // TODO: we might be able to change the last conditional so that
    // fluctuations only apply if the endpoint energy is *greater* than the
    // tracking cut (eloss_calc_limit). That could allows us to skip the
    // boundary check.
    if (fluct_params_ && eloss > zero_quantity() && eloss < particle.energy())
    {
        // Apply energy loss fluctuations
        auto cutoffs = track.make_cutoff_view();
        auto material = track.make_material_view();

        EnergyLossHelper loss_helper(
            fluct_params_, cutoffs, material, particle, eloss, step);

        auto rng = track.make_rng_engine();
        switch (loss_helper.model())
        {
#define ASU_SAMPLE_ELOSS(MODEL)                                              \
    case EnergyLossFluctuationModel::MODEL:                                  \
        eloss = this->sample_energy_loss<EnergyLossFluctuationModel::MODEL>( \
            loss_helper, particle.energy(), rng);                            \
        break
            ASU_SAMPLE_ELOSS(none);
            ASU_SAMPLE_ELOSS(gamma);
            ASU_SAMPLE_ELOSS(gaussian);
            ASU_SAMPLE_ELOSS(urban);
#undef ASU_SAMPLE_ELOSS
        }
    }
    CELER_ASSERT(eloss <= particle.energy());
    return eloss;
}

//---------------------------------------------------------------------------//
template<EnergyLossFluctuationModel M>
CELER_FUNCTION auto
FluctELoss::sample_energy_loss(EnergyLossHelper const& helper,
                               Energy max_loss,
                               RngEngine& rng) -> Energy
{
    CELER_EXPECT(helper.model() == M);

    using Distribution = typename EnergyLossTraits<M>::type;

    Distribution sample_eloss{helper};
    Energy result = sample_eloss(rng);

    // TODO: investigate cases where sampled energy loss is greater than
    // the track's actual energy, i.e. the range limiter failed.
    result = Energy{celeritas::min(result.value(), max_loss.value())};
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

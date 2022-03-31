//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanMscHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/ArrayUtils.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/grid/EnergyLossCalculator.hh"
#include "physics/grid/InverseRangeCalculator.hh"
#include "physics/grid/RangeCalculator.hh"
#include "physics/material/Types.hh"
#include "random/distributions/NormalDistribution.hh"

#include "UrbanMscData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * This is a helper class for the UrbanMscStepLimit and UrbanMscScatter.
 */
class UrbanMscHelper
{
  public:
    //!@{
    //! Type aliases
    using Energy       = units::MevEnergy;
    using MaterialData = detail::UrbanMscMaterialData;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscHelper(const UrbanMscNativeRef& shared,
                                         const ParticleTrackView& particle,
                                         const PhysicsTrackView&  physics,
                                         const MaterialView&      material);

    //// COMMON PROPERTIES ////

    //! A scale factor for the range
    static CELER_CONSTEXPR_FUNCTION real_type dtrl() { return 5e-2; }

    //! The lower bound of energy to scale the mininum true path length limit
    static CELER_CONSTEXPR_FUNCTION Energy tlow() { return Energy(5e-3); }

    //! The minimum value of the true path length limit (0.01*CLHEP::nm)
    static CELER_CONSTEXPR_FUNCTION real_type limit_min_fix()
    {
        return 1e-9 * units::centimeter;
    }

    //// HELPER FUNCTIONS ////

    //! The step length from physics processes
    CELER_FUNCTION real_type step_length() const
    {
        return physics_.step_length();
    }

    //! The range for a given particle energy
    CELER_FUNCTION real_type range() const { return range_; }

    // The mean free path of the multiple scattering for a given energy
    inline CELER_FUNCTION real_type msc_mfp(Energy energy) const;

    // The total energy loss over a given step length
    inline CELER_FUNCTION Energy eloss(real_type step) const;

    // The kinetic energy at the end of a given step length corrected by dedx
    inline CELER_FUNCTION Energy end_energy(real_type step) const;

    // Calculate the minimum of the true path length limit
    inline CELER_FUNCTION real_type calc_limit_min(Energy    energy,
                                                   real_type step_min) const;

    // Calculate the minimum of the step length for a given elastic mfp
    inline CELER_FUNCTION real_type calc_step_min(Energy    energy,
                                                  real_type lambda) const;

    // Sample a random true path length limit
    template<class Engine>
    inline CELER_FUNCTION real_type randomize_limit(Engine&   rng,
                                                    real_type limit,
                                                    real_type limit_min) const;

    // Calculate the true path length from the geom path length
    inline CELER_FUNCTION real_type calc_true_path(real_type true_path,
                                                   real_type geom_path,
                                                   real_type alpha) const;

  private:
    //// DATA ////

    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle flag for positron
    const bool is_positron_;
    // PhysicsTrackView
    const PhysicsTrackView& physics_;
    // Material dependent data
    const MaterialData& msc_;
    // Shared value of range
    real_type range_;
    // Shared value of small tau
    real_type tau_small_;
    // Process ID of the eletromagnetic_msc process
    ParticleProcessId msc_pid_;
    // Grid ID of range value of the energy loss
    ValueGridId range_gid_;
    // Grid ID of dedx value of the energy loss
    ValueGridId eloss_gid_;
    // Grid ID of the lambda value of MSC
    ValueGridId mfp_gid_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscHelper::UrbanMscHelper(const UrbanMscNativeRef& shared,
                               const ParticleTrackView& particle,
                               const PhysicsTrackView&  physics,
                               const MaterialView&      material)
    : inc_energy_(particle.energy())
    , is_positron_(particle.particle_id() == shared.positron_id)
    , physics_(physics)
    , msc_(shared.msc_data[material.material_id()])
    , tau_small_(shared.params.tau_small)
{
    CELER_EXPECT(particle.particle_id() == shared.electron_id
                 || particle.particle_id() == shared.positron_id);

    ParticleProcessId eloss_pid = physics.eloss_ppid();
    range_gid_ = physics.value_grid(ValueGridType::range, eloss_pid);
    eloss_gid_ = physics.value_grid(ValueGridType::energy_loss, eloss_pid);
    msc_pid_   = physics_.msc_ppid();
    mfp_gid_   = physics_.value_grid(ValueGridType::msc_mfp, msc_pid_);
    range_ = physics.make_calculator<RangeCalculator>(range_gid_)(inc_energy_);
}

//// HELPER FUNCTIONS ////

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean free path of the msc for a given particle energy.
 */
CELER_FUNCTION real_type UrbanMscHelper::msc_mfp(Energy energy) const
{
    real_type xsec = physics_.calc_xs(msc_pid_, mfp_gid_, energy)
                     / ipow<2>(energy.value());
    // Return the mean free path
    real_type mfp = (xsec > 0) ? 1 / xsec
                               : numeric_limits<real_type>::infinity();
    return mfp;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the total energy loss over a given step length.
 */
CELER_FUNCTION auto UrbanMscHelper::eloss(real_type step) const -> Energy
{
    auto calc_energy
        = physics_.make_calculator<InverseRangeCalculator>(range_gid_);

    return calc_energy(step);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the kinetic energy at the end of a given msc step.
 */
CELER_FUNCTION auto UrbanMscHelper::end_energy(real_type step) const -> Energy
{
    // The energy loss per unit length for a given particle energy
    real_type dedx = physics_.make_calculator<EnergyLossCalculator>(
        eloss_gid_)(inc_energy_);

    return (step > range_ * this->dtrl())
               ? this->eloss(range_ - step)
               : Energy{inc_energy_.value() - step * dedx};
}

//---------------------------------------------------------------------------//
/*!
 * Define the minimum step using the ratio of lambda_elastic/lambda_transport.
 */
CELER_FUNCTION real_type UrbanMscHelper::calc_step_min(Energy    energy,
                                                       real_type lambda) const
{
    real_type re = energy.value();

    return lambda
           / (2 + real_type(1e3) * re * (msc_.stepmin_a + msc_.stepmin_b * re));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the minimum of the true path length limit.
 */
CELER_FUNCTION real_type UrbanMscHelper::calc_limit_min(Energy    energy,
                                                        real_type step) const
{
    real_type xm = (is_positron_)
                       ? real_type(0.70) * step * std::sqrt(msc_.zeff)
                       : real_type(0.87) * step * msc_.z23;

    if (energy.value() < this->tlow().value())
    {
        // Energy is below a pre-defined limit
        xm *= real_type(0.5) * (1 + energy.value() / this->tlow().value());
    }

    return max<real_type>(xm, this->limit_min_fix());
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random true path length limit.
 */
template<class Engine>
CELER_FUNCTION real_type UrbanMscHelper::randomize_limit(
    Engine& rng, real_type limit, real_type limit_min) const
{
    real_type result = limit_min;
    if (limit > limit_min)
    {
        NormalDistribution<real_type> gauss(
            limit, real_type(0.1) * (limit - limit_min));
        result = gauss(rng);
        result = max<real_type>(result, limit_min);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Compute the true path length for a given geom path (the z -> t conversion).
 *
 * The transformation can be written as
 * \f[
 *     t(z) = \langle t \rangle = -\lambda_{1} \log(1 - \frac{z}{\lambda_{1}})
 * \f]
 * or \f$ t(z) = \frac{1}{\alpha} [ 1 - (1-\alpha w z)^{1/w}] \f$ if the
 * geom path is small, where \f$ w = 1 + \frac{1}{\alpha \lambda_{10}}\f$.
 *
 * @param true_path the proposed step before transportation.
 * @param geom_path the proposed step after transportation.
 * @param alpha variable from UrbanMscStepLimit.
 */
CELER_FUNCTION
real_type UrbanMscHelper::calc_true_path(real_type true_path,
                                         real_type geom_path,
                                         real_type alpha) const
{
    if (geom_path < 100 * this->limit_min_fix())
    {
        return geom_path;
    }

    // Recalculation
    real_type lambda = msc_mfp(inc_energy_);
    real_type length = geom_path;

    // NOTE: add && !insideskin if the UseDistanceToBoundary algorithm is used
    if (geom_path > lambda * tau_small_)
    {
        if (alpha < 0)
        {
            // For cases that the true path is very small compared to either
            // the mean free path or the range
            length = -lambda * std::log(1 - geom_path / lambda);
        }
        else
        {
            real_type w = 1 + 1 / (alpha * lambda);
            real_type x = alpha * w * geom_path;
            length      = (x < 1) ? (1 - std::exp(std::log(1 - x) / w)) / alpha
                                  : range_;
        }

        length = min<real_type>(true_path, max<real_type>(geom_path, length));
    }

    return length;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

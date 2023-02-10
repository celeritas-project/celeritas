//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/UrbanMscStepLimit.hh
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
#include "celeritas/random/Selector.hh"
#include "celeritas/random/distribution/NormalDistribution.hh"

#include "UrbanMscHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * This is the step limitation algorithm of the Urban model for the e-/e+
 * multiple scattering.

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
    using Mass = units::MevMass;
    using MscParameters = UrbanMscParameters;
    using MaterialData = UrbanMscMaterialData;
    using UrbanMscRef = NativeCRef<UrbanMscData>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscStepLimit(UrbanMscRef const& shared,
                                            ParticleTrackView const& particle,
                                            PhysicsTrackView* physics,
                                            MaterialId matid,
                                            bool on_boundary,
                                            real_type safety,
                                            real_type phys_step);

    // Apply the step limitation algorithm for the e-/e+ MSC with the RNG
    template<class Engine>
    inline CELER_FUNCTION MscStep operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant data
    UrbanMscRef const& shared_;
    // PhysicsTrackView
    PhysicsTrackView& physics_;
    // Incident particle energy [Energy]
    const real_type inc_energy_;
    // Incident particle safety
    const real_type safety_;
    // Urban MSC setable parameters
    MscParameters const& params_;
    // Urban MSC material-dependent data
    MaterialData const& msc_;
    // Urban MSC helper class
    UrbanMscHelper helper_;
    // Urban MSC range properties
    MscRange msc_range_;

    // Transport mean free path
    real_type lambda_{};
    // Physics step length
    real_type phys_step_{};
    // Mean slowing-down distance from current energy to zero
    real_type range_{};
    // Whether to skip sampling and just return the original physics step
    bool skip_displacement_{false};

    //// HELPER TYPES ////

    struct GeomPathAlpha
    {
        real_type geom_path;
        real_type alpha;
    };

    //// HELPER FUNCTIONS ////

    // Calculate the geometry path length for a given true path length
    inline CELER_FUNCTION GeomPathAlpha calc_geom_path(real_type true_path) const;

    // Calculate the minimum of the true path length limit
    inline CELER_FUNCTION real_type calc_limit_min() const;

    //! The lower bound of energy to scale the mininum true path length limit
    static CELER_CONSTEXPR_FUNCTION Energy tlow()
    {
        return units::MevEnergy(5e-3);
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscStepLimit::UrbanMscStepLimit(UrbanMscRef const& shared,
                                     ParticleTrackView const& particle,
                                     PhysicsTrackView* physics,
                                     MaterialId matid,
                                     bool on_boundary,
                                     real_type safety,
                                     real_type phys_step)
    : shared_(shared)
    , physics_(*physics)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , safety_(safety)
    , params_(shared.params)
    , msc_(shared_.material_data[matid])
    , helper_(shared, particle, physics_)
    , msc_range_(physics_.msc_range())
    , lambda_{helper_.calc_msc_mfp(Energy{inc_energy_})}
    , phys_step_(phys_step)
    , range_(physics_.dedx_range())
{
    CELER_EXPECT(particle.particle_id() == shared.ids.electron
                 || particle.particle_id() == shared.ids.positron);
    CELER_EXPECT(safety_ >= 0);
    CELER_EXPECT(lambda_ > 0);
    CELER_EXPECT(phys_step > 0);
    CELER_EXPECT(phys_step_ <= range_);

    if (phys_step_ < shared_.params.limit_min_fix())
    {
        // Very short step: don't displace
        skip_displacement_ = true;
    }
    else if (range_ * msc_.d_over_r < safety_)
    {
        // Potential step length is shorter than potential boundary distance
        // NOTE: use d_over_r_mh for muons and charged hadrons
        skip_displacement_ = true;
    }
    else if (!msc_range_ || on_boundary)
    {
        // Initialize MSC range cache on the first step in a volume
        msc_range_.range_fact = params_.range_fact;
        msc_range_.range_init = max<real_type>(range_, lambda_);
        if (lambda_ > params_.lambda_limit)
        {
            msc_range_.range_fact
                *= (real_type(0.75)
                    + real_type(0.25) * lambda_ / params_.lambda_limit);
        }
        msc_range_.limit_min = this->calc_limit_min();

        // Store persistent range properties within this tracking volume
        physics_.msc_range(msc_range_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the true path length using the Urban multiple scattering model
 * as well as the geometry path length for a given proposed physics step.
 *
 * The
 * model is selected for the candidate process governing the step if the true
 * path length is smaller than the current physics step length. However, the
 * geometry path length will be used for the further step length competition
 * (either with the linear or field propagator). If the geometry path length
 * is smaller than the distance to the next volume, then the model is finally
 * selected for the interaction by the multiple scattering.
 */
template<class Engine>
CELER_FUNCTION auto UrbanMscStepLimit::operator()(Engine& rng) -> MscStep
{
    MscStep result;
    result.true_path = phys_step_;

    // The case for a very small step or the lower limit for the linear
    // distance that e-/e+ can travel is far from the geometry boundary
    if (skip_displacement_)
    {
        MscStep result;
        result.is_displaced = false;
        result.true_path = phys_step_;
        result.geom_path = this->calc_geom_path(phys_step_).geom_path;
        return result;
    }

    // Step limitation algorithm: UseSafety (the default)
    // TODO: add options for other algorithms (see G4MscStepLimitType.hh)

    // The step limit
    real_type limit = range_;
    if (limit > safety_)
    {
        limit = max<real_type>(msc_range_.range_fact * msc_range_.range_init,
                               params_.safety_fact * safety_);
    }
    limit = max<real_type>(limit, msc_range_.limit_min);

    if (limit < result.true_path)
    {
        // Randomize the limit if this step should be determined by msc
        real_type sampled_limit = msc_range_.limit_min;
        if (limit > sampled_limit)
        {
            NormalDistribution<real_type> sample_gauss(
                limit, real_type(0.1) * (limit - msc_range_.limit_min));
            sampled_limit = sample_gauss(rng);
            sampled_limit = max<real_type>(sampled_limit, msc_range_.limit_min);
        }
        result.true_path = min<real_type>(result.true_path, sampled_limit);
    }

    {
        auto temp = this->calc_geom_path(result.true_path);
        result.geom_path = temp.geom_path;
        result.alpha = temp.alpha;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the geometry step length for a given true step length.
 *
 * The mean value of the geometrical path length \f$ z \f$ (the first moment)
 * corresponding to a given true path length \f$ t \f$ is given by
 * \f[
 *     \langle z \rangle = \lambda_{1} [ 1 - \exp({-\frac{t}{\lambda_{1}}})]
 * \f]
 * where \f$\lambda_{1}\f$ is the first transport mean free path. Due to the
 * fact that \f$\lambda_{1}\f$ depends on the kinetic energy of the path and
 * decreases along the step, the path length correction is approximated as
 * \f[
 *     \lambda_{1} (t) = \lambda_{10} (1 - \alpha t)
 * \f]
 * where \f$ \alpha = \frac{\lambda_{10} - \lambda_{11}}{t\lambda_{10}} \f$
 * or  \f$ \alpha = 1/r_0 \f$ in a simpler form with the range \f$ r_0 \f$
 * if the kinetic energy of the particle is below its mass -
 * \f$ \lambda_{10} (\lambda_{11}) \f$ denotes the value of \f$\lambda_{1}\f$
 * at the start (end) of the step, respectively.
 *
 * \note This performs the same method as in ComputeGeomPathLength of
 * G4UrbanMscModel of the Geant4 10.7 release.
 */
CELER_FUNCTION
auto UrbanMscStepLimit::calc_geom_path(real_type true_path) const
    -> GeomPathAlpha
{
    // Do the true path -> geom path transformation
    GeomPathAlpha result;
    result.geom_path = true_path;
    result.alpha = MscStep::tiny_step_alpha();

    if (true_path < shared_.params.min_step())
    {
        // geometrical path length = true path length for a very small step
        return result;
    }

    // tau = number of mean free paths to collision
    real_type tau = true_path / lambda_;
    if (tau <= params_.tau_small)
    {
        // Very small distance to collision
        result.geom_path = min<real_type>(true_path, lambda_);
    }
    else if (true_path < range_ * shared_.params.dtrl())
    {
        // XXX use expm1 instead?
        result.geom_path = (tau < params_.tau_limit)
                               ? true_path * (1 - tau / 2)
                               : lambda_ * (1 - std::exp(-tau));
    }
    else if (inc_energy_ < value_as<Mass>(shared_.electron_mass)
             || true_path == range_)
    {
        // Low energy or range-limited step
        result.alpha = 1 / range_;
        real_type w = 1 + 1 / (result.alpha * lambda_);

        result.geom_path = 1 / (result.alpha * w);
        if (true_path < range_)
        {
            result.geom_path *= (1 - fastpow(1 - true_path / range_, w));
        }
    }
    else
    {
        // Calculate the energy at the end of a physics-limited step
        real_type rfinal
            = max<real_type>(range_ - true_path, real_type(0.01) * range_);
        Energy endpoint_energy = helper_.calc_stopping_energy(rfinal);
        real_type lambda1 = helper_.calc_msc_mfp(endpoint_energy);

        // Calculate the geometric path assuming the cross section is linear
        // between the start and end energy.
        // Eq 8.10+1
        result.alpha = (lambda_ - lambda1) / (lambda_ * true_path);
        CELER_ASSERT(result.alpha != MscStep::tiny_step_alpha());
        // Eq 8.10 with simplifications
        real_type w = 1 + 1 / (result.alpha * lambda_);
        result.geom_path = (1 - fastpow(lambda1 / lambda_, w))
                           / (result.alpha * w);
    }

    result.geom_path = min<real_type>(result.geom_path, lambda_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the minimum of the true path length limit.
 */
CELER_FUNCTION real_type UrbanMscStepLimit::calc_limit_min() const
{
    using PolyQuad = PolyEvaluator<real_type, 2>;

    // Calculate minimum step
    real_type xm = lambda_
                   / PolyQuad(2, msc_.stepmin_a, msc_.stepmin_b)(inc_energy_);

    // Scale based on particle type and effective atomic number
    xm *= helper_.scaled_zeff();

    if (inc_energy_ < value_as<Energy>(this->tlow()))
    {
        // Energy is below a pre-defined limit
        xm *= (real_type(0.5)
               + real_type(0.5) * inc_energy_ / value_as<Energy>(this->tlow()));
    }

    return max<real_type>(xm, shared_.params.limit_min_fix());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UrbanMscStepLimit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/material/Types.hh"
#include "random/Selector.hh"
#include "sim/SimTrackView.hh"

#include "UrbanMscData.hh"
#include "UrbanMscHelper.hh"

namespace celeritas
{
namespace detail
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
    //! Type aliases
    using Energy        = units::MevEnergy;
    using MscParameters = detail::UrbanMscParameters;
    using MaterialData  = detail::UrbanMscMaterialData;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscStepLimit(const UrbanMscRef&       shared,
                                            const ParticleTrackView& particle,
                                            GeoTrackView*            geometry,
                                            const PhysicsTrackView&  physics,
                                            const MaterialView&      material,
                                            const SimTrackView&      sim);

    // Apply the step limitation algorithm for the e-/e+ MSC with the RNG
    template<class Engine>
    inline CELER_FUNCTION MscStep operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant data
    const UrbanMscRef& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle safety
    const real_type safety_;
    // Urban MSC setable parameters
    const MscParameters& params_;
    // Urban MSC material-dependent data
    const MaterialData& msc_;
    // Urban MSC helper class
    UrbanMscHelper helper_;

    bool      on_boundary_{};
    real_type range_{};
    real_type lambda_{};

    //// HELPER TYPES ////

    struct GeomPathAlpha
    {
        real_type geom_path;
        real_type alpha;
    };

    //// HELPER FUNCTIONS ////

    // Calculate the geometry path length for a given true path length
    inline CELER_FUNCTION GeomPathAlpha calc_geom_path(real_type true_path) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscStepLimit::UrbanMscStepLimit(const UrbanMscRef&       shared,
                                     const ParticleTrackView& particle,
                                     GeoTrackView*            geometry,
                                     const PhysicsTrackView&  physics,
                                     const MaterialView&      material,
                                     const SimTrackView&      sim)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , safety_(geometry->find_safety(geometry->pos()))
    , params_(shared.params)
    , msc_(shared_.msc_data[material.material_id()])
    , helper_(shared, particle, physics, material)
    , on_boundary_(sim.num_steps() == 0 || safety_ == 0)
{
    CELER_EXPECT(particle.particle_id() == shared.ids.electron
                 || particle.particle_id() == shared.ids.positron);

    range_  = helper_.range();
    lambda_ = helper_.msc_mfp(inc_energy_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the true path length using the Urban multiple scattering model
 * as well as the geometry path length for a given proposed physics step. The
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

    result.phys_step = helper_.step_length();
    result.true_path = min<real_type>(result.phys_step, range_);

    // The case for a very small step or the lower limit for the linear
    // distance that e-/e+ can travel is far from the geometry boundary
    // NOTE: use d_over_r_mh for muons and charged hadrons
    real_type distance = range_ * msc_.d_over_r;
    if (result.true_path < helper_.limit_min_fix()
        || (safety_ > 0 && distance < safety_))
    {
        result.is_displaced = false;
        auto temp           = this->calc_geom_path(result.true_path);
        result.geom_path    = temp.geom_path;
        return result;
    }

    // Step limitation algorithm: UseSafety (the default)
    // TODO: add options for other algorithms (see G4MscStepLimitType.hh)

    // Initialisation at the first step or at the boundary
    real_type range_fact = params_.range_fact;
    real_type range_init = max<real_type>(range_, lambda_);

    // G4StepStatus = fGeomBoundary: step defined by a geometry boundary
    if (on_boundary_)
    {
        // For the first step of a track or after entering in a new volume
        if (lambda_ > params_.lambda_limit)
        {
            range_fact *= (real_type(0.75)
                           + real_type(0.25) * lambda_ / params_.lambda_limit);
        }
        real_type step_min = helper_.calc_step_min(inc_energy_, lambda_);
        result.limit_min   = helper_.calc_limit_min(inc_energy_, step_min);
    }

    // The step limit
    real_type limit = range_;
    if (limit > safety_)
    {
        limit = max<real_type>(range_fact * range_init,
                               params_.safety_fact * safety_);
    }
    limit = max<real_type>(limit, result.limit_min);

    if (limit < result.true_path)
    {
        // Randomize the limit if this step should be determined by msc
        result.true_path = min<real_type>(
            result.true_path,
            helper_.randomize_limit(rng, limit, result.limit_min));
    }

    {
        auto temp        = this->calc_geom_path(result.true_path);
        result.geom_path = temp.geom_path;
        result.alpha     = temp.alpha;
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
 * or  \f$ \alpha = 1/r_0 \f$ in a simpler form with the range \f$ r_o \f$
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
    result.alpha     = -1;

    if (true_path < 100 * helper_.limit_min_fix())
    {
        // geometrical path length = true path length for a very small step
        return result;
    }

    real_type tau = true_path / lambda_;
    if (tau <= params_.tau_small)
    {
        result.geom_path = min<real_type>(true_path, lambda_);
    }
    else if (true_path < range_ * helper_.dtrl())
    {
        result.geom_path = (tau < params_.tau_limit)
                               ? true_path * (1 - tau / 2)
                               : lambda_ * (1 - std::exp(-tau));
    }
    else if (inc_energy_.value() < shared_.electron_mass.value()
             || true_path == range_)
    {
        result.alpha = 1 / range_;
        real_type w  = 1 + 1 / (result.alpha * lambda_);

        result.geom_path = 1 / (result.alpha * w);
        if (true_path < range_)
        {
            result.geom_path
                *= (1 - std::exp(w * std::log(1 - true_path / range_)));
        }
    }
    else
    {
        real_type rfinal
            = max<real_type>(range_ - true_path, real_type(0.01) * range_);
        Energy    loss    = helper_.eloss(rfinal);
        real_type lambda1 = helper_.msc_mfp(loss);

        result.alpha     = (lambda_ - lambda1) / (lambda_ * true_path);
        real_type w      = 1 + 1 / (result.alpha * lambda_);
        result.geom_path = (1 - std::exp(w * std::log(lambda1 / lambda_)))
                           / (result.alpha * w);
    }

    result.geom_path = min<real_type>(result.geom_path, lambda_);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

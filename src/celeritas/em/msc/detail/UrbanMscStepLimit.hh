//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
 * This distribution is to be used for tracks that have nonnegligble steps and
 * are near the boundary. Otherwise, no displacement or step limiting is
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
    using Mass = units::MevMass;
    using MscParameters = UrbanMscParameters;
    using MaterialData = UrbanMscMaterialData;
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
    // Incident particle energy [Energy]
    Energy const inc_energy_;
    // Incident particle safety
    real_type const safety_;
    // Urban MSC material-dependent data
    MaterialData const& msc_;
    // Urban MSC range properties
    MscRange const& msc_range_;

    // Physics step length
    real_type phys_step_{};
    // Mean slowing-down distance from current energy to zero
    real_type range_{};

    //// HELPER TYPES ////

    struct GeomPathAlpha
    {
        real_type geom_path;
        real_type alpha;
    };

    //// HELPER FUNCTIONS ////

    // Calculate the minimum of the true path length limit
    inline CELER_FUNCTION real_type calc_limit_min() const;
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
    : shared_(shared)
    , helper_(helper)
    , inc_energy_(inc_energy)
    , safety_(safety)
    , msc_(shared_.material_data[matid])
    , msc_range_(physics->msc_range())
    , phys_step_(phys_step)
    , range_(physics->dedx_range())
{
    CELER_EXPECT(safety_ >= 0);
    CELER_EXPECT(safety_ < helper_.max_step());
    CELER_EXPECT(phys_step_ > shared_.params.limit_min_fix());
    CELER_EXPECT(phys_step_ <= range_);
    if (!msc_range_ || on_boundary)
    {
        MscRange new_range;
        // Initialize MSC range cache on the first step in a volume
        // TODO for hadrons/muons: this value is hard-coded for electrons
        new_range.range_fact = shared.params.range_fact;
        // XXX the 1 MFP limitation is applied to the *geo* step, not the true
        // step, so this isn't quite right (See UrbanMsc.hh)
        new_range.range_init = max<real_type>(range_, helper_.msc_mfp());
        if (helper_.msc_mfp() > shared.params.lambda_limit)
        {
            new_range.range_fact *= (real_type(0.75)
                                     + real_type(0.25) * helper_.msc_mfp()
                                           / shared.params.lambda_limit);
        }
        new_range.limit_min = this->calc_limit_min();

        // Store persistent range properties within this tracking volume
        physics->msc_range(new_range);
        // Range is a reference so should be updated
        CELER_ASSERT(msc_range_);
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
CELER_FUNCTION auto UrbanMscStepLimit::operator()(Engine& rng) -> real_type
{
    // Step limitation algorithm: UseSafety (the default)
    real_type limit = range_;
    if (safety_ < range_)
    {
        limit = max<real_type>(msc_range_.range_fact * msc_range_.range_init,
                               shared_.params.safety_fact * safety_);
    }
    limit = max<real_type>(limit, msc_range_.limit_min);

    real_type true_path = phys_step_;
    if (limit < phys_step_)
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
        true_path = min<real_type>(phys_step_, sampled_limit);
    }

    return true_path;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the minimum of the true path length limit.
 */
CELER_FUNCTION real_type UrbanMscStepLimit::calc_limit_min() const
{
    using PolyQuad = PolyEvaluator<real_type, 2>;

    // Calculate minimum step
    PolyQuad calc_min_mfp(2, msc_.stepmin_coeff[0], msc_.stepmin_coeff[1]);
    real_type xm = helper_.msc_mfp() / calc_min_mfp(inc_energy_.value());

    // Scale based on particle type and effective atomic number
    xm *= helper_.scaled_zeff();

    if (inc_energy_ < shared_.params.min_scaling_energy())
    {
        // Energy is below a pre-defined limit
        xm *= (real_type(0.5)
               + real_type(0.5) * value_as<Energy>(inc_energy_)
                     / value_as<Energy>(shared_.params.min_scaling_energy()));
    }

    return max<real_type>(xm, shared_.params.limit_min_fix());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

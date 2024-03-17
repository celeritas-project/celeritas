//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/detail/UrbanMscHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/grid/EnergyLossCalculator.hh"
#include "celeritas/grid/InverseRangeCalculator.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/grid/ValueGridData.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * This is a helper class for the UrbanMscStepLimit and UrbanMscScatter.
 *
 * NOTE: units are "native" units, listed here as CGS.
 *
 * \todo Refactor to UrbanMscTrackView .
 */
class UrbanMscHelper
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using MaterialData = UrbanMscMaterialData;
    using UrbanMscRef = NativeCRef<UrbanMscData>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION UrbanMscHelper(UrbanMscRef const& shared,
                                         ParticleTrackView const& particle,
                                         PhysicsTrackView const& physics);

    //// HELPER FUNCTIONS ////

    //! The mean free path of the multiple scattering at the current energy
    //! [len]
    CELER_FUNCTION real_type msc_mfp() const { return lambda_; }

    // The mean free path of the multiple scattering for a given energy [len]
    inline CELER_FUNCTION real_type calc_msc_mfp(Energy energy) const;

    // TODO: the following methods are used only by MscStepLimit

    // Calculate the energy corresponding to a given particle range
    inline CELER_FUNCTION Energy calc_inverse_range(real_type step) const;

    //! Step limit scaling based on atomic number and particle type
    CELER_FUNCTION real_type scaled_zeff() const
    {
        return this->pmdata().scaled_zeff;
    }

    // Maximum expected distance based on the track's range
    inline CELER_FUNCTION real_type max_step() const;

    // The kinetic energy at the end of a given step length corrected by dedx
    inline CELER_FUNCTION Energy calc_end_energy(real_type step) const;

  private:
    //// DATA ////

    // References to external data
    UrbanMscRef const& shared_;
    ParticleTrackView const& particle_;
    PhysicsTrackView const& physics_;

    // Precalculated mean free path (TODO: move to physics step view)
    real_type lambda_;  // [len]

    // Data for this particle+material
    CELER_FUNCTION UrbanMscParMatData const& pmdata() const
    {
        return shared_.par_mat_data[shared_.at<UrbanMscParMatData>(
            physics_.material_id(), particle_.particle_id())];
    }

    // Scaled cross section data for this particle+material
    CELER_FUNCTION XsGridData const& xs() const
    {
        return shared_.xs[shared_.at<XsGridData>(physics_.material_id(),
                                                 particle_.particle_id())];
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
UrbanMscHelper::UrbanMscHelper(UrbanMscRef const& shared,
                               ParticleTrackView const& particle,
                               PhysicsTrackView const& physics)
    : shared_(shared)
    , particle_(particle)
    , physics_(physics)
    , lambda_(this->calc_msc_mfp(particle_.energy()))
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.electron
                 || particle.particle_id() == shared_.ids.positron);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean free path of the msc for a given particle energy.
 */
CELER_FUNCTION real_type UrbanMscHelper::calc_msc_mfp(Energy energy) const
{
    CELER_EXPECT(energy > zero_quantity());
    XsCalculator calc_scaled_xs(this->xs(), shared_.reals);

    real_type xsec = calc_scaled_xs(energy) / ipow<2>(energy.value());
    CELER_ENSURE(xsec >= 0 && 1 / xsec > 0);
    return 1 / xsec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy corresponding to a given particle range.
 *
 * This is an exact value based on the range claculation. It can be used to
 * find the exact energy loss over a step.
 */
CELER_FUNCTION auto UrbanMscHelper::calc_inverse_range(real_type step) const
    -> Energy
{
    auto range_gid
        = physics_.value_grid(ValueGridType::range, physics_.eloss_ppid());
    auto range_to_energy
        = physics_.make_calculator<InverseRangeCalculator>(range_gid);
    return range_to_energy(step);
}

//---------------------------------------------------------------------------//
/*!
 * Maximum expected step length based on the track's range.
 */
CELER_FUNCTION real_type UrbanMscHelper::max_step() const
{
    return physics_.dedx_range() * this->pmdata().d_over_r;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the kinetic energy at the end of a given msc step.
 */
CELER_FUNCTION auto UrbanMscHelper::calc_end_energy(real_type step) const
    -> Energy
{
    CELER_EXPECT(step <= physics_.dedx_range());
    real_type range = physics_.dedx_range();
    if (step <= range * shared_.params.dtrl())
    {
        auto eloss_gid = physics_.value_grid(ValueGridType::energy_loss,
                                             physics_.eloss_ppid());
        // Assume constant energy loss rate over the step
        real_type dedx = physics_.make_calculator<EnergyLossCalculator>(
            eloss_gid)(particle_.energy());

        return particle_.energy() - Energy{step * dedx};
    }
    else
    {
        // Longer step is calculated exactly with inverse range
        return this->calc_inverse_range(range - step);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

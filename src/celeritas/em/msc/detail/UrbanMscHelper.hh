//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "celeritas/grid/UniformLogGridCalculator.hh"
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

    // Data for this particle+material
    inline CELER_FUNCTION UrbanMscParMatData const& pmdata() const;

    // Scaled cross section data for this particle+material
    inline CELER_FUNCTION UniformGridRecord const& xs() const;

  private:
    //// DATA ////

    // References to external data
    UrbanMscRef const& shared_;
    ParticleTrackView const& particle_;
    PhysicsTrackView const& physics_;

    // Precalculated mean free path (TODO: move to physics step view)
    real_type lambda_;  // [len]
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
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean free path of the msc for a given particle energy.
 */
CELER_FUNCTION real_type UrbanMscHelper::calc_msc_mfp(Energy energy) const
{
    CELER_EXPECT(energy > zero_quantity());
    UniformLogGridCalculator calc_scaled_xs(this->xs(), shared_.reals);

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
    auto range_to_energy = physics_.make_calculator<InverseRangeCalculator>(
        physics_.inverse_range_grid());
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
    if (step <= range * shared_.params.small_range_frac)
    {
        // Assume constant energy loss rate over the step
        real_type dedx = physics_.make_calculator<EnergyLossCalculator>(
            physics_.energy_loss_grid())(particle_.energy());

        return particle_.energy() - Energy{step * dedx};
    }
    else
    {
        // Longer step is calculated exactly with inverse range
        return this->calc_inverse_range(range - step);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Scaled cross section data for this particle+material.
 */
CELER_FUNCTION UniformGridRecord const& UrbanMscHelper::xs() const
{
    auto par_id = shared_.pid_to_xs[particle_.particle_id()];
    CELER_ASSERT(par_id < shared_.num_particles);

    size_type idx = physics_.material_id().get() * shared_.num_particles
                    + par_id.unchecked_get();
    CELER_ASSERT(idx < shared_.xs.size());

    return shared_.xs[ItemId<UniformGridRecord>(idx)];
}

//---------------------------------------------------------------------------//
/*!
 * Data for this particle+material.
 */
CELER_FUNCTION UrbanMscParMatData const& UrbanMscHelper::pmdata() const
{
    auto par_id = shared_.pid_to_pmdata[particle_.particle_id()];
    CELER_ASSERT(par_id < shared_.num_par_mat);

    size_type idx = physics_.material_id().get() * shared_.num_par_mat
                    + par_id.unchecked_get();
    CELER_ASSERT(idx < shared_.par_mat_data.size());

    return shared_.par_mat_data[ItemId<UrbanMscParMatData>(idx)];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

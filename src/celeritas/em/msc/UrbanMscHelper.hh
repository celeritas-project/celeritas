//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/UrbanMscHelper.hh
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
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * This is a helper class for the UrbanMscStepLimit and UrbanMscScatter.
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

    // The mean free path of the multiple scattering for a given energy
    inline CELER_FUNCTION real_type calc_msc_mfp(Energy energy) const;

    // TODO: the following methods are used only by MscStepLimit

    // The total energy loss over a given step length
    inline CELER_FUNCTION Energy calc_stopping_energy(real_type step) const;

    // The kinetic energy at the end of a given step length corrected by dedx
    inline CELER_FUNCTION Energy calc_end_energy(real_type step) const;

    //! Step limit scaling based on atomic number and particle type
    CELER_FUNCTION real_type scaled_zeff() const
    {
        return pmdata_.scaled_zeff;
    }

  private:
    //// DATA ////

    // Incident particle energy
    const real_type inc_energy_;
    // PhysicsTrackView
    PhysicsTrackView const& physics_;
    // Range scaling factor
    const real_type dtrl_;

    // Grid ID of range value of the energy loss
    ValueGridId range_gid_;
    // Grid ID of dedx value of the energy loss
    ValueGridId eloss_gid_;

    // Data for this particle + material
    UrbanMscParMatData const& pmdata_;
    // Calculate the transport cross section with a factor of E^2 built in
    XsCalculator calc_scaled_xs_;
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
    : inc_energy_(value_as<Energy>(particle.energy()))
    , physics_(physics)
    , dtrl_(shared.params.dtrl())
    , pmdata_(shared.par_mat_data[shared.at(physics.material_id(),
                                            particle.particle_id())])
    , calc_scaled_xs_(pmdata_.xs, shared.reals)
{
    CELER_EXPECT(particle.particle_id() == shared.ids.electron
                 || particle.particle_id() == shared.ids.positron);

    ParticleProcessId eloss_pid = physics.eloss_ppid();
    range_gid_ = physics.value_grid(ValueGridType::range, eloss_pid);
    eloss_gid_ = physics.value_grid(ValueGridType::energy_loss, eloss_pid);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean free path of the msc for a given particle energy.
 */
CELER_FUNCTION real_type UrbanMscHelper::calc_msc_mfp(Energy energy) const
{
    real_type xsec = calc_scaled_xs_(energy) / ipow<2>(energy.value());
    CELER_ENSURE(xsec >= 0);
    return 1 / xsec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy of a track that is stopped after the given step.
 *
 * This is an exact value based on the range claculation.
 */
CELER_FUNCTION auto UrbanMscHelper::calc_stopping_energy(real_type step) const -> Energy
{
    auto range_to_energy
        = physics_.make_calculator<InverseRangeCalculator>(range_gid_);

    return range_to_energy(step);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the kinetic energy at the end of a given msc step.
 */
CELER_FUNCTION auto UrbanMscHelper::calc_end_energy(real_type step) const
    -> Energy
{
    real_type range = physics_.dedx_range();
    if (step <= range * dtrl_)
    {
        // Short step can be approximated with linear extrapolation.
        real_type dedx = physics_.make_calculator<EnergyLossCalculator>(
            eloss_gid_)(Energy{inc_energy_});

        return Energy{inc_energy_ - step * dedx};
    }
    else
    {
        // Longer step is calculated exactly with inverse range
        return this->calc_stopping_energy(range - step);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

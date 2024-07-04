//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/IoniFinalStateHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Quantities.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Update final state of incident particle and delta ray for ionization.
 */
class IoniFinalStateHelper
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION IoniFinalStateHelper(Energy inc_energy,
                                               Real3 const& inc_direction,
                                               Momentum inc_momentum,
                                               Mass inc_mass,
                                               Energy electron_energy,
                                               Mass electron_mass,
                                               ParticleId electron_id,
                                               Secondary* secondary);

    // Update the final state of the incident particle and secondary electron
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Incident particle energy [MeV]
    real_type inc_energy_;
    // Incident particle direction
    Real3 const& inc_direction_;
    // Incident particle momentum [MeV / c]
    real_type inc_momentum_;
    // Incident particle mass
    real_type inc_mass_;
    // Secondary electron energy [MeV]
    real_type electron_energy_;
    // Electron mass
    real_type electron_mass_;
    // Secondary electron ID
    ParticleId electron_id_;
    // Allocated secondary electron
    Secondary* secondary_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and exiting gamma data.
 */
CELER_FUNCTION
IoniFinalStateHelper::IoniFinalStateHelper(Energy inc_energy,
                                           Real3 const& inc_direction,
                                           Momentum inc_momentum,
                                           Mass inc_mass,
                                           Energy electron_energy,
                                           Mass electron_mass,
                                           ParticleId electron_id,
                                           Secondary* secondary)
    : inc_energy_(value_as<Energy>(inc_energy))
    , inc_direction_(inc_direction)
    , inc_momentum_(value_as<Momentum>(inc_momentum))
    , inc_mass_(value_as<Mass>(inc_mass))
    , electron_energy_(value_as<Energy>(electron_energy))
    , electron_mass_(value_as<Mass>(electron_mass))
    , electron_id_(electron_id)
    , secondary_{secondary}
{
    CELER_EXPECT(secondary_);
}

//---------------------------------------------------------------------------//
/*!
 * Update the final state of the incident particle and secondary electron.
 */
template<class Engine>
CELER_FUNCTION Interaction IoniFinalStateHelper::operator()(Engine& rng)
{
    // Calculate the polar angle of the exiting electron
    real_type momentum = std::sqrt(electron_energy_
                                   * (electron_energy_ + 2 * electron_mass_));
    real_type costheta = electron_energy_
                         * (inc_energy_ + inc_mass_ + electron_mass_)
                         / (momentum * inc_momentum_);
    CELER_ASSERT(costheta <= 1);

    // Sample and save outgoing secondary data
    secondary_->energy = Energy{electron_energy_};
    secondary_->direction
        = ExitingDirectionSampler{costheta, inc_direction_}(rng);
    secondary_->particle_id = electron_id_;

    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.energy = Energy{inc_energy_ - electron_energy_};
    result.direction = calc_exiting_direction(
        {inc_momentum_, inc_direction_}, {momentum, secondary_->direction});
    result.secondaries = {secondary_, 1};

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

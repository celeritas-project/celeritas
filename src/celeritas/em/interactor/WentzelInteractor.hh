//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/WentzelInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/em/distribution/WentzelDistribution.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/IsotopeSelector.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Applies the Wentzel single Coulomb scattering model.
 *
 * This models incident high-energy electrons and positrons elastically
 * scattering off of nuclei and atomic electrons. Scattering off of the nucleus
 * versus electrons is randomly sampled based on the relative cross-sections.
 * No secondaries are created in this process (in the future, with hadronic
 * transport support, secondary ions may be emitted), however production cuts
 * are used to determine the maximum scattering angle off of electrons.
 *
 * \note This performs the same sampling as in Geant4's
 *  G4eCoulombScatteringModel, as documented in section 8.2 of the Geant4
 *  Physics Reference Manual (release 11.1).
 */
class WentzelInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using MomentumSq = units::MevMomentumSq;
    //!@}

  public:
    //! Construct with shared and state data
    inline CELER_FUNCTION WentzelInteractor(WentzelRef const& shared,
                                            ParticleTrackView const& particle,
                                            Real3 const& inc_direction,
                                            MaterialView const& material,
                                            ElementComponentId const& elcomp_id,
                                            CutoffView const& cutoffs);

    //! Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Constant shared data
    WentzelRef const& data_;

    // Incident direction
    Real3 const& inc_direction_;

    // Incident particle
    ParticleTrackView const& particle_;

    // Mott coefficients of the target element
    WentzelElementData const& element_data_;

    // Material's cutoff energy for the incident particle
    real_type const cutoff_energy_;

    // Target element
    ElementView const element_;

    //// HELPER FUNCTIONS ////

    //! Calculates the recoil energy for the given scattering direction
    inline CELER_FUNCTION real_type calc_recoil_energy(
        Real3 const& new_direction, Mass const& target_mass) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared and state data
 */
CELER_FUNCTION
WentzelInteractor::WentzelInteractor(WentzelRef const& shared,
                                     ParticleTrackView const& particle,
                                     Real3 const& inc_direction,
                                     MaterialView const& material,
                                     ElementComponentId const& elcomp_id,
                                     CutoffView const& cutoffs)
    : data_(shared)
    , inc_direction_(inc_direction)
    , particle_(particle)
    , element_data_(shared.elem_data[material.element_id(elcomp_id)])
    , cutoff_energy_(value_as<Energy>(cutoffs.energy(particle.particle_id())))
    , element_(material.make_element_view(elcomp_id))
{
    CELER_EXPECT(particle_.particle_id() == data_.ids.electron
                 || particle_.particle_id() == data_.ids.positron);
    CELER_EXPECT(particle_.energy() > detail::coulomb_scattering_limit()
                 && particle_.energy() < detail::high_energy_limit());
}

//---------------------------------------------------------------------------//
/*!
 * Sample the Coulomb scattering of the incident particle.
 */
template<class Engine>
CELER_FUNCTION Interaction WentzelInteractor::operator()(Engine& rng)
{
    // Select an isotope of the target nucleus
    IsotopeSelector iso_select(element_);
    IsotopeView target = element_.make_isotope_view(iso_select(rng));

    // Distribution model governing the scattering
    WentzelDistribution sample_direction(
        particle_, target, element_data_, cutoff_energy_, data_);

    // Incident particle scatters
    Interaction result;

    // Sample the new direction
    Real3 new_direction = sample_direction(rng);
    result.direction = rotate(inc_direction_, new_direction);

    // Recoil energy is kinetic energy transfered to the atom
    real_type inc_energy = value_as<Energy>(particle_.energy());
    real_type recoil_energy
        = clamp(calc_recoil_energy(new_direction, target.nuclear_mass()),
                real_type{0},
                inc_energy);
    result.energy = Energy{inc_energy - recoil_energy};

    // TODO: For high enough recoil energies, ions are produced

    result.energy_deposition = Energy{recoil_energy};

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the recoil energy for the given scattering direction calculated
 * by WentzelDistribution.
 */
CELER_FUNCTION real_type WentzelInteractor::calc_recoil_energy(
    Real3 const& new_direction, Mass const& target_mass) const
{
    real_type one_minus_cos_theta = 1 - new_direction[2];
    return value_as<MomentumSq>(particle_.momentum_sq()) * one_minus_cos_theta
           / (value_as<Mass>(target_mass)
              + (value_as<Mass>(particle_.mass())
                 + value_as<Energy>(particle_.energy()))
                    * one_minus_cos_theta);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

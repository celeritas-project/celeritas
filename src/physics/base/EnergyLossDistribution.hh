//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyLossDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simulate energy loss fluctuations.
 *
 * The energy loss of charged particles is a stochastic quantity with a
 * distribution described by the straggling function. Fluctuations in the
 * energy loss over a given thickness of material arise from statistical
 * variation in both the number of collisions and the energy lost in each
 * collision. Above a given energy threshold, the fluctuations in the energy
 * loss are simulated by the explicit sampling of secondaries. However, the
 * continuous energy loss below the cutoff energy also has fluctuations, and
 * these are not taken into account in the calculation of the mean loss. Energy
 * loss fluctuation models are used to sample the actual restricted energy loss
 * given the mean loss.
 *
 * Different models are used depending on the value of the parameter \f$ \kappa
 * = \xi / T_{max} \f$, the ratio of the mean energy loss to the maximum
 * allowed energy transfer in a single collision. For large \f$ \kappa \f$,
 * when the particle loses all or most of its energy in the absorber, the
 * number of collisions is large and a Gaussian approximation of the straggling
 * function can be used. Otherwise, the Urban model for energy loss
 * fluctuations in thin layers is applied.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4UniversalFluctuation, as documented in section 7.3 in the Geant4 Physics
 * Reference Manual, PHYS332 and PHYS333 in GEANT3, CERN Program Library Long
 * Writeup, W5013 (1993), and K. Lassila-Perini and L. Urban, Nucl. Instr. and
 * Meth. in Phys. Res. A 362 (1995) 416-422.
 */
class EnergyLossDistribution
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    using Real2     = Array<real_type, 2>;
    //!@}

  public:
    // Construct from material, incident particle, and mean energy loss
    inline CELER_FUNCTION
    EnergyLossDistribution(const MaterialView&      material,
                           const ParticleTrackView& particle,
                           const CutoffView&        cutoffs,
                           ParticleId               electron_id,
                           MevEnergy                mean_loss,
                           real_type                step_length);

    // Calculate the actual energy loss from the mean loss
    template<class Engine>
    inline CELER_FUNCTION MevEnergy operator()(Engine& rng) const;

  private:
    //// DATA ////

    // Shared properties of the current material
    const MaterialView& material_;
    // Average energy loss calculated from the tables
    const real_type mean_loss_;
    // Distance over which the incident particle lost the energy
    const real_type step_length_;
    // Ratio of electron mass to incident particle mass
    const real_type mass_ratio_;
    // Incident particle charge
    const real_type charge_;
    // Incident particle Lorentz factor
    const real_type gamma_;
    // Square of the Lorentz factor
    const real_type gamma_sq_;
    // Square of the ratio of the particle velocity to the speed of light
    const real_type beta_sq_;
    // Maximum possible energy transfer to an electron in a single collision
    const real_type max_energy_transfer_;
    // Smaller of the delta ray production cut and maximum energy transfer
    const real_type max_energy_;

    //// CONSTANTS ////

    //! Electron mass in MeV/c^2
    static CELER_CONSTEXPR_FUNCTION real_type electron_mass()
    {
        return constants::electron_mass / unit_cast(units::MevMass{1});
    }

    //! Minimum mean energy loss required to sample fluctuations
    static CELER_CONSTEXPR_FUNCTION MevEnergy min_valid_energy()
    {
        return MevEnergy{1e-5};
    }

    //! Atomic energy level corresponding to outer electrons (E_0)
    static CELER_CONSTEXPR_FUNCTION MevEnergy ionization_energy()
    {
        return MevEnergy{1e-5};
    }

    //! Lower limit on the number of interactions in a step (kappa)
    static CELER_CONSTEXPR_FUNCTION size_type min_kappa() { return 10; }

    //! Relative contribution of ionization to energy loss
    static CELER_CONSTEXPR_FUNCTION real_type rate() { return 0.56; }

    //! Number of collisions above which to use faster sampling from Gaussian
    static CELER_CONSTEXPR_FUNCTION size_type max_collisions() { return 8; }

    //! Threshold number of excitations used in width correction
    static CELER_CONSTEXPR_FUNCTION real_type exc_thresh() { return 42; }

    //// HELPER FUNCTIONS ////

    template<class Engine>
    inline CELER_FUNCTION real_type sample_gaussian(Engine& rng) const;

    template<class Engine>
    inline CELER_FUNCTION real_type sample_urban(Engine& rng) const;

    template<class Engine>
    inline CELER_FUNCTION real_type sample_excitation_loss(Real2 xs,
                                                           Real2 binding_energy,
                                                           Engine& rng) const;

    template<class Engine>
    inline CELER_FUNCTION real_type sample_ionization_loss(real_type xs,
                                                           Engine& rng) const;

    template<class Engine>
    inline CELER_FUNCTION real_type sample_fast_urban(real_type mean,
                                                      real_type stddev,
                                                      Engine&   rng) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "EnergyLossDistribution.i.hh"

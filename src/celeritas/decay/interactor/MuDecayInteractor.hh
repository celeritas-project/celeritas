//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/interactor/MuDecayInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/decay/data/MuDecayData.hh"
#include "celeritas/phys/FourVector.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform muon decay.
 *
 * Only one decay channel is implemented, with muons decaying to
 * \f[
 * \mu^- \longrightarrow e^- \overline{\nu}_e \nu_\mu
 * \f]
 * or
 * \f[
 * \mu^+ \longrightarrow e^+ \nu_e \overline{\nu}_\mu
 * \f].
 *
 * This interactor follows \c G4MuonDecayChannel and the Physics Reference
 * Manual, Release 11.2, section 4.2.3.
 *
 * \warning Neutrinos are currently not returned in this interactor as they
 * are not tracked down or transported.
 *
 * \note The full three-body decay can be recovered from PR #1456, commit
 * `ecc4326`.
 */
class MuDecayInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using MevMomentum = units::MevMomentum;
    using Mass = units::MevMass;
    using UniformRealDist = UniformRealDistribution<real_type>;
    //!@}

    // Construct with shared and state data
    inline CELER_FUNCTION
    MuDecayInteractor(MuDecayData const& shared,
                      ParticleTrackView const& particle,
                      Real3 const& inc_direction,
                      StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Constant data
    MuDecayData const& shared_;
    // Incident muon energy
    Energy const inc_energy_;
    // Incident muon direction
    Real3 const& inc_direction_;
    // Allocate space for secondary particles (currently, electron only)
    StackAllocator<Secondary>& allocate_;
    // Define decay channel based on muon or anti-muon primary
    bool is_muon_;
    // Incident muon four vector
    FourVector inc_fourvec_;
    // Max sampled fractional energy for the electron neutrino
    real_type sample_max_;
    // Maximum electron energy
    real_type max_energy_;

    //// HELPER FUNCTIONS ////

    // Boost four vector from the rest frame to the lab frame
    inline CELER_FUNCTION FourVector to_lab_frame(Real3 const& dir,
                                                  MevMomentum const& momentum,
                                                  Mass const& mass) const;

    // Calculate particle momentum (or kinetic energy) in the center of mass
    inline CELER_FUNCTION MevMomentum calc_momentum(real_type const& energy_frac,
                                                    Mass const& mass) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
MuDecayInteractor::MuDecayInteractor(MuDecayData const& shared,
                                     ParticleTrackView const& particle,
                                     Real3 const& inc_direction,
                                     StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.mu_minus
                 || particle.particle_id() == shared_.ids.mu_plus);

    // Define decay channel
    is_muon_ = (particle.particle_id() == shared_.ids.mu_minus) ? true : false;

    // Set up muon four vector to boost decay to the lab-frame
    inc_fourvec_ = {inc_direction_ * particle.momentum().value(),
                    particle.total_energy().value()};

    // Sampling constants
    sample_max_
        = real_type{1}
          + ipow<2>(shared_.electron_mass.value() / shared_.muon_mass.value());

    // Geant4 physics manual defines E_{max} = m_\mu / 2, while the source code
    // (since v10.2.0 at least) defines E_{max} = m_\mu / 2 - m_e . The source
    // code implementation leads to a total CM energy of ~104.6 MeV instead of
    // the expected 105.7 MeV (muon mass), which is achieved by using the
    // physics manual definition
    max_energy_ = real_type{0.5} * shared_.muon_mass.value()
                  - shared_.electron_mass.value();
}

//---------------------------------------------------------------------------//
/*!
 * Sample the muon decay.
 *
 * The decay is sampled at the muon rest frame (based on
 * \c G4MuonDecayChannel::DecayIt ) and the resulting four vector is boosted to
 * the lab frame using the muon's incident energy and direction.
 *
 * Since only the charged lepton is being returned, these steps are ommitted:
 * - Calculate the angle between charged lepton and electron neutrino.
 * - Calculate the energies of the neutrinos.
 * - Calculate the final directions of the neutrinos using the aforementioned
 * angle and conservation of momentum.
 * - Sample a spherically uniform direction and rotate all three secondaries
 * using `EulerRotation`.
 * - Boost all particles to the lab frame
 */
template<class Engine>
CELER_FUNCTION Interaction MuDecayInteractor::operator()(Engine& rng)
{
    // Allocate secondaries
    Secondary* secondaries = allocate_(1);
    if (secondaries == nullptr)
    {
        // Failed to allocate secondaries
        return Interaction::from_failure();
    }

    real_type electron_energy_frac{};
    real_type electron_nu_energy_frac{};
    do
    {
        do
        {
            electron_nu_energy_frac = UniformRealDist(0, sample_max_)(rng);
        } while (generate_canonical(rng)
                 > electron_nu_energy_frac
                       * (real_type{1} - electron_nu_energy_frac));

        electron_energy_frac = generate_canonical(rng);
    } while (electron_nu_energy_frac < real_type{1} - electron_energy_frac);

    // Momentum of secondaries at rest frame
    auto charged_lep_energy
        = this->calc_momentum(electron_energy_frac, shared_.electron_mass);

    // Apply a spherically uniform rotation to the decay
    IsotropicDistribution rotate;
    Real3 charged_lep_dir = rotate(rng);

    // Boost secondaries to the lab frame
    auto charged_lep_4vec = this->to_lab_frame(
        charged_lep_dir, charged_lep_energy, shared_.electron_mass);

    // Return electron only
    Interaction result = Interaction::from_decay();
    result.secondaries = {secondaries, 1};
    result.secondaries[0].particle_id = is_muon_ ? shared_.ids.electron
                                                 : shared_.ids.positron;
    // Interaction stores kinetic energy; FourVector stores total energy
    result.secondaries[0].energy
        = Energy{charged_lep_4vec.energy - shared_.electron_mass.value()};
    result.secondaries[0].direction = make_unit_vector(charged_lep_4vec.mom);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Boost secondary to the lab frame.
 *
 * \note This assumes the primary to be at rest and, thus, there is no need
 * to perform an inverse boost of the primary at the CM frame.
 */
CELER_FUNCTION FourVector MuDecayInteractor::to_lab_frame(
    Real3 const& dir, MevMomentum const& momentum, Mass const& mass) const
{
    CELER_EXPECT(norm(dir) > 0);
    CELER_EXPECT(momentum > zero_quantity());
    CELER_EXPECT(mass >= zero_quantity());

    Real3 p = dir * momentum.value();
    FourVector lepton_4vec{
        p, std::sqrt(ipow<2>(norm(p)) + ipow<2>(mass.value()))};
    boost(boost_vector(inc_fourvec_), &lepton_4vec);

    return lepton_4vec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate final particle momentum (or kinetic energy) from its sampled
 * fractional energy.
 */
CELER_FUNCTION units::MevMomentum
MuDecayInteractor::calc_momentum(real_type const& energy_frac,
                                 Mass const& mass) const
{
    return MevMomentum{
        std::sqrt(ipow<2>(energy_frac * max_energy_)
                  + 2 * energy_frac * max_energy_ * mass.value())};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

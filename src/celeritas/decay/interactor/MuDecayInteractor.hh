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
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/EulerRotation.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/decay/data/MuDecayData.hh"
#include "celeritas/phys/FourVector.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
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
 */
class MuDecayInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
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
    MuDecayData shared_;
    // Incident muon energy
    units::MevEnergy const inc_energy_;
    // Incident muon direction
    Real3 const inc_direction_;
    // Allocate space for a secondary particles
    StackAllocator<Secondary>& allocate_;
    // Define decay channel based on muon or anti-muon primary
    ParticleId secondary_id_[3];
    // Incident muon four vector
    FourVector inc_fourvec_;
    // Max sampled fractional energy
    real_type sample_max_;
    // Maximum electron energy
    real_type max_energy_;

    //// HELPER FUNCTIONS ////

    // Boost four vector from the rest frame to the lab frame
    inline CELER_FUNCTION FourVector to_lab_frame(Real3 const& dir,
                                                  Energy const& energy,
                                                  real_type const& mass);

    // Calculate particle momentum (or kinetic energy) in the center of mass
    inline CELER_FUNCTION Energy momentum(real_type const& energy_frac,
                                          real_type const& mass);
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
    auto const& ids = shared_.ids;
    bool muon = (particle.particle_id() == ids.mu_minus) ? true : false;
    secondary_id_[0] = muon ? ids.electron : ids.positron;
    secondary_id_[1] = muon ? ids.anti_electron_neutrino
                            : ids.electron_neutrino;
    secondary_id_[2] = muon ? ids.muon_neutrino : ids.anti_muon_neutrino;

    // Set up muon four vector to boost decay to the lab-frame
    Real3 inc_momentum = inc_direction_ * inc_energy_.value();
    inc_fourvec_ = {
        inc_momentum,
        std::sqrt(ipow<2>(norm(inc_momentum)) + ipow<2>(shared_.muon_mass))};

    // Sampling constants
    sample_max_ = real_type{1}
                  + ipow<2>(shared_.electron_mass) / ipow<2>(shared_.muon_mass);

    // Geant4 physics manual defines E_{max} = m_\mu / 2, while the source code
    // (since v10.2.0 at least) defines E_{max} = m_\mu / 2 - m_e . The source
    // code implementation leads to a total CM energy of ~104.6 MeV instead of
    // the expected 105.7 MeV (muon mass), which is achieved by using the
    // physics manual definition
    max_energy_ = 0.5 * shared_.muon_mass;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a muon decay via the
 * \f[
 * \mu^- \longrightarrow e^- \overline{\nu}_e \nu_\mu
 * \f]
 * or
 * \f[
 * \mu^+ \longrightarrow e^+ \nu_e \overline{\nu}_\mu
 * \f]
 * channel, with a branching ratio of 100%.
 *
 * The decay is sampled at the muon's reference frame (based on
 * \c G4MuonDecayChannel::DecayIt ) and the resulting four vector is boosted to
 * the lab frame using the muon's incident energy and direction.
 *
 * \note Neutrinos are assumed to have zero mass.
 */
template<class Engine>
CELER_FUNCTION Interaction MuDecayInteractor::operator()(Engine& rng)
{
    // Allocate secondaries
    Secondary* secondaries = allocate_(3);
    if (secondaries == nullptr)
    {
        // Failed to allocate secondaries
        return Interaction::from_failure();
    }

    real_type electron_energy_frac;
    real_type electron_nu_energy_frac;
    size_type const max_iter{1000};  // \todo Why?

    for ([[maybe_unused]] auto i : range(max_iter))
    {
        for ([[maybe_unused]] auto j : range(max_iter))
        {
            electron_nu_energy_frac = sample_max_ * generate_canonical(rng);
            if (generate_canonical(rng)
                <= electron_nu_energy_frac
                       * (real_type{1} - electron_nu_energy_frac))
            {
                break;
            }
            electron_nu_energy_frac = sample_max_;
        }

        electron_energy_frac = generate_canonical(rng);
        if (electron_nu_energy_frac >= real_type{1} - electron_energy_frac)
        {
            break;
        }
        electron_nu_energy_frac = real_type{1} - electron_energy_frac;
    }
    real_type muon_nu_energy_frac = real_type{2} - electron_energy_frac
                                    - electron_nu_energy_frac;

    // Momentum of secondaries at rest frame; neutrino masses are ignored
    auto charged_lep_energy
        = this->momentum(electron_energy_frac, shared_.electron_mass);
    auto electron_nu_energy = this->momentum(electron_nu_energy_frac, 0);
    auto muon_nu_energy = this->momentum(muon_nu_energy_frac, 0);

    // Angle between charged lepton and electron neutrino
    real_type cos_theta
        = real_type{1} - 2 / electron_energy_frac - 2 / electron_nu_energy_frac
          + 2 / (electron_nu_energy_frac * electron_energy_frac);
    CELER_ASSERT(std::fabs(cos_theta) <= 1);
    real_type sin_theta = std::sqrt(real_type{1} - ipow<2>(cos_theta));

    // Use an initial arbitrary direction for secondaries.
    // Placing the electron on one axis (e.g. +z), simplifies the calculation
    // for the other two secondaries. The direction of the electron neutrino
    // is defined by the sampled theta, and the muon neutrino direction is
    // calculated via conservation of momentum
    Real3 charged_lep_dir = {0, 0, 1};
    Real3 electron_nu_dir = {sin_theta, 0, cos_theta};
    Real3 muon_nu_dir
        = {-(electron_nu_energy_frac / muon_nu_energy_frac) * sin_theta,
           0,
           -electron_energy_frac / muon_nu_energy_frac
               - (electron_nu_energy_frac / muon_nu_energy_frac) * cos_theta};

    // Apply a spherically uniform rotation to the decay
    real_type euler_phi
        = UniformRealDistribution<real_type>(0, 2 * constants::pi)(rng);
    real_type euler_theta = std::acos(2 * generate_canonical(rng) - 1);
    real_type euler_psi
        = UniformRealDistribution<real_type>(0, 2 * constants::pi)(rng);

    EulerRotation rotate(euler_phi, euler_theta, euler_psi);
    charged_lep_dir = rotate(charged_lep_dir);
    electron_nu_dir = rotate(electron_nu_dir);
    muon_nu_dir = rotate(muon_nu_dir);

    // Boost secondaries to the lab frame
    auto charged_lep_4vec = this->to_lab_frame(
        charged_lep_dir, charged_lep_energy, shared_.electron_mass);
    auto electron_nu_4vec
        = this->to_lab_frame(electron_nu_dir, electron_nu_energy, 0);
    auto muon_nu_4vec = this->to_lab_frame(muon_nu_dir, muon_nu_energy, 0);

    // \todo Return electron only?
    Interaction result = Interaction::from_absorption();
    result.action = Interaction::Action::decay;
    result.secondaries = {secondaries, 3};

    // Interaction stores kinetic energy; FourVector stores total energy
    result.secondaries[0].particle_id = secondary_id_[0];
    result.secondaries[0].energy
        = Energy{charged_lep_4vec.energy - shared_.electron_mass};
    result.secondaries[0].direction = make_unit_vector(charged_lep_4vec.mom);

    result.secondaries[1].particle_id = secondary_id_[1];
    result.secondaries[1].energy = Energy{electron_nu_4vec.energy};  // no mass
    result.secondaries[1].direction = make_unit_vector(electron_nu_4vec.mom);

    result.secondaries[2].particle_id = secondary_id_[2];
    result.secondaries[2].energy = Energy{muon_nu_4vec.energy};  // no mass
    result.secondaries[2].direction = make_unit_vector(muon_nu_4vec.mom);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Boost secondary to the lab frame.
 *
 * \note This assumes the primary to be at rest and, thus, there is no need to
 * perform an inverse boost of the primary at the CM frame.
 */
CELER_FUNCTION FourVector MuDecayInteractor::to_lab_frame(Real3 const& dir,
                                                          Energy const& energy,
                                                          real_type const& mass)
{
    CELER_EXPECT(norm(dir) > 0);
    CELER_EXPECT(energy > zero_quantity());
    CELER_EXPECT(mass >= 0);

    Real3 p = dir * energy.value();  // Momentum [MeV]
    FourVector lepton_4vec{p, std::sqrt(ipow<2>(norm(p)) + ipow<2>(mass))};
    boost(boost_vector(inc_fourvec_), &lepton_4vec);

    return lepton_4vec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate final particle momentum (or kinetic energy) from its sampled
 * fractional energy.
 */
CELER_FUNCTION units::MevEnergy
MuDecayInteractor::momentum(real_type const& energy_frac, real_type const& mass)
{
    return Energy{std::sqrt(ipow<2>(energy_frac) * ipow<2>(max_energy_)
                            + 2 * energy_frac * max_energy_ * mass)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

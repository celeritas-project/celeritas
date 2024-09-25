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
 */
class MuDecayInteractor
{
  public:
    //!@{
    // \name Type aliases
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
    // Incident direction
    Real3 const& inc_direction_;
    // Secondary electron or positron id
    ParticleId secondary_id_;
    // Allocate space for a secondary particles
    StackAllocator<Secondary>& allocate_;
    // Inverse boost of incident muon
    Real3 inv_boost_inc_fourvec_;
    // Max possible sampled energy
    real_type max_energy_;
    // Max sampled fractional energy
    real_type sample_max_;

    //// HELPER FUNCTIONS ////

    // Return boosted four vector to the lab frame
    inline CELER_FUNCTION FourVector to_lab_frame(Real3 const& dir,
                                                  Energy const& energy,
                                                  real_type const& mass);

    // Return rotated final direction in the muon's rest frame
    inline CELER_FUNCTION Real3 calc_cm_dir(Real3 const& dir,
                                            real_type const& costheta,
                                            real_type const& phi);

    // Calculate final particle energ from the sampled fraction
    inline CELER_FUNCTION Energy calc_energy(real_type const& energy_frac,
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

    secondary_id_ = (particle.particle_id() == shared_.ids.mu_minus)
                        ? shared_.ids.electron
                        : shared_.ids.positron;

    // \todo Set up whole decay channel if we do return the neutrinos

    // Inverse boost incident muon four vector for later use
    Real3 inc_momentum = inc_direction_ * inc_energy_.value();
    FourVector inc_4vec{
        inc_momentum,
        std::sqrt(ipow<2>(norm(inc_momentum)) + ipow<2>(shared_.muon_mass))};
    inv_boost_inc_fourvec_ = -boost_vector(inc_4vec);

    // \todo This differs from physics manual, that states E_{max} = m_\mu / 2
    max_energy_ = 0.5 * shared_.muon_mass - shared_.electron_mass;

    sample_max_ = real_type{1}
                  + ipow<2>(shared_.electron_mass) / ipow<2>(shared_.muon_mass);
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
 * The decay is sampled at the muon's reference frame and the resulting four
 * vector is boosted to the lab frame using the muon's incident energy and
 * direction.
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

    real_type sample_result;
    real_type electron_nu_energy_frac;
    real_type sampled_energy_frac;  // Outgoing electron

    size_type const max_iter = 1000;  // \todo Why?

    for ([[maybe_unused]] auto i : range(max_iter))
    {
        for ([[maybe_unused]] auto j : range(max_iter))
        {
            sample_result = sample_max_ * generate_canonical(rng);
            real_type rej = generate_canonical(rng);
            if (rej <= sample_result * (real_type{1} - sample_result))
            {
                break;
            }
            sample_result = sample_max_;
        }

        sampled_energy_frac = generate_canonical(rng);  // Charged lepton
        electron_nu_energy_frac = sampled_energy_frac;
        if (sample_result >= real_type{1} - sampled_energy_frac)
        {
            break;
        }
        electron_nu_energy_frac = real_type{1} - sampled_energy_frac;
    }

    real_type muon_nu_energy_frac = real_type{2} - sampled_energy_frac
                                    - electron_nu_energy_frac;

    // Angle between charged lepton and electron neutrino
    real_type cos_theta = real_type{1} - 0.5 * sample_result
                          - 0.5 * electron_nu_energy_frac
                          + 0.5 * (electron_nu_energy_frac * sample_result);
    CELER_ASSERT(std::fabs(cos_theta) <= 1);
    real_type sin_theta = std::sqrt(1 - ipow<2>(cos_theta));
    CELER_ASSERT(std::fabs(sin_theta) <= 1);

    // Sample spherically uniform direction to be applied to the decay
    real_type decay_dir_costheta
        = std::cos(2 * constants::pi * generate_canonical(rng));
    real_type decay_dir_phi = constants::pi * generate_canonical(rng);

    // Charged lepton
    auto charged_lep_energy
        = this->calc_energy(sample_result, shared_.electron_mass);
    // Start from a random dir (e.g. +z) and rotate it
    auto charged_lep_dir
        = this->calc_cm_dir({0, 0, 1}, decay_dir_costheta, decay_dir_phi);

    // Electron neutrino
    auto electron_nu_energy = this->calc_energy(electron_nu_energy_frac, 0);
    auto electron_nu_dir = this->calc_cm_dir(
        {sin_theta, 0, cos_theta}, decay_dir_costheta, decay_dir_phi);

    // Muon neutrino
    auto muon_nu_energy = this->calc_energy(muon_nu_energy_frac, 0);
    auto muon_nu_dir = this->calc_cm_dir(
        {-sin_theta * electron_nu_energy_frac / muon_nu_energy_frac,
         0,
         -sample_result / muon_nu_energy_frac
             - cos_theta * electron_nu_energy_frac / muon_nu_energy_frac},
        decay_dir_costheta,
        decay_dir_phi);

    // Move all three particles to lab frame
    auto charged_lep_4vec = this->to_lab_frame(
        charged_lep_dir, charged_lep_energy, shared_.electron_mass);
    auto electron_nu_4vec
        = this->to_lab_frame(electron_nu_dir, electron_nu_energy, 0);
    auto muon_nu_4vec = this->to_lab_frame(muon_nu_dir, muon_nu_energy, 0);

    // \todo Return electron only?
    Interaction result;
    result.action = Interaction::Action::decay;
    result.secondaries = {secondaries, 3};

    result.secondaries[0].particle_id = secondary_id_;
    result.secondaries[0].energy = Energy{charged_lep_4vec.energy};
    result.secondaries[0].direction = make_unit_vector(charged_lep_4vec.mom);

    result.secondaries[1].energy = Energy{electron_nu_4vec.energy};
    result.secondaries[1].direction = make_unit_vector(electron_nu_4vec.mom);

    result.secondaries[2].energy = Energy{muon_nu_4vec.energy};
    result.secondaries[2].direction = make_unit_vector(muon_nu_4vec.mom);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Boost secondary to the lab frame
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

    boost(inv_boost_inc_fourvec_, &lepton_4vec);
    boost(boost_vector(lepton_4vec), &lepton_4vec);

    return lepton_4vec;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate decay direction for particle in the muon's rest frame.
 */
CELER_FUNCTION Real3 MuDecayInteractor::calc_cm_dir(Real3 const& dir,
                                                    real_type const& costheta,
                                                    real_type const& phi)
{
    Real3 result = make_unit_vector(dir);
    result = rotate(result, from_spherical(costheta, phi));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate final particle energy (or momentum) from the sampled fractional
 * energy with respect to the maximum possible energy ( \c max_energy_ ).
 */
CELER_FUNCTION units::MevEnergy
MuDecayInteractor::calc_energy(real_type const& energy_frac,
                               real_type const& mass)
{
    return Energy{std::sqrt(ipow<2>(energy_frac) * ipow<2>(max_energy_)
                            + 2 * energy_frac * max_energy_ * mass)};
    ;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

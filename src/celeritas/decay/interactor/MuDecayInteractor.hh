//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/random/distribution/RejectionSampler.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/decay/data/MuDecayData.hh"
#include "celeritas/phys/FourVector.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform muon decay.
 *
 * Only one decay channel is implemented, with muons decaying to
 * \f[
 * \mu^- \longrightarrow e^- \bar{\nu}_e \nu_\mu
 * \f]
 * or
 * \f[
 * \mu^+ \longrightarrow e^+ \nu_e \bar{\nu}_\mu
 * \f].
 *
 * This interactor follows \c G4MuonDecayChannel::DecayIt and the Physics
 * Reference Manual, Release 11.2, section 4.2.3. The sampling happens at the
 * muon's rest frame, with the result being uniformly rotated and finally
 * boosted to the lab frame.
 *
 * As it is a three-body decay, the energy sampling happens for \f$ e^\pm \f$
 * and \f$ \nu_e(\bar{\nu}_e) \f$, with the \f$ \nu_\mu(\bar{\nu}_\mu) \f$
 * final energy directly calculated from energy conservation. The sampling loop
 * selects fractional energies \f$[0,1)\f$ for the first two particles. A
 * fraction of 1 yields the maximum possible kinetic energy for said particle,
 * defined as \f$ E_\text{max} = \frac{m_\mu}{2} - m_e \f$. For the electron
 * neutrino, its energy fraction \f$ f_{E_{\nu_e}} \f$ is sampled from the PDF
 * \f$ f(x) = 6x(1-x), \ x \in [0,1) \f$, using the rejection method, with
 * proposal distribution PDF \f$ g(x) = U(0,1) \f$ and bounding constant
 * \f$ M = 1.5 \f$. The charged lepton's fractional energy \f$ f_{E_e} \f$ is
 * then selected uniformly obeying \f$ f_{E_{\nu_e}} + f_{E_e} >= 1 \f$ . The
 * remaining fractional energy for the muon neutrino is
 * \f$ f_{E_{\nu_\mu}} = 2 - f_{E_e} - f_{E_{\nu_e}} \f$.
 *
 * \note Neutrinos are currently not returned by this interactor as they are
 * not tracked down or transported and would significantly increase secondary
 * memory allocation usage. See discussion and commit history at
 * https://github.com/celeritas-project/celeritas/pull/1456
 */
class MuDecayInteractor
{
  public:
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
    //// TYPES ////

    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    using Mass = units::MevMass;

    //// DATA ////

    // Constant data
    MuDecayData const& shared_;
    // Incident muon energy
    Energy const inc_energy_;
    // Allocate space for secondary particles (electron only)
    StackAllocator<Secondary>& allocate_;
    // Define decay channel based on muon or anti-muon primary
    ParticleId sec_id_;
    // Incident muon four vector
    FourVector inc_fourvec_;
    // Maximum electron energy [MeV]
    real_type max_energy_;

    //// HELPER FUNCTIONS ////

    // Boost four vector from the rest frame to the lab frame
    inline CELER_FUNCTION FourVector to_lab_frame(Real3 const& dir,
                                                  Momentum momentum,
                                                  Mass mass) const;

    // Calculate particle momentum (or kinetic energy) in the center of mass
    inline CELER_FUNCTION Momentum calc_momentum(real_type energy_frac,
                                                 Mass mass) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * \note Geant4 physics manual defines \f$ E_{max} = m_\mu / 2 \f$,
 * while the source code (since v10.2.0 at least) defines
 * \f$ E_{max} = m_\mu / 2 - m_e \f$ .
 * The source code implementation leads to a total CM energy of ~104.6
 * MeV instead of the expected 105.7 MeV (muon mass), which is achieved by
 * using the physics manual definition.
 */
CELER_FUNCTION
MuDecayInteractor::MuDecayInteractor(MuDecayData const& shared,
                                     ParticleTrackView const& particle,
                                     Real3 const& inc_direction,
                                     StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , allocate_(allocate)
    , sec_id_((particle.particle_id() == shared_.mu_minus_id)
                  ? shared_.electron_id
                  : shared_.positron_id)
    , inc_fourvec_{FourVector::from_particle(particle, inc_direction)}
    , max_energy_(real_type{0.5} * shared_.muon_mass.value()
                  - shared_.electron_mass.value())
{
    CELER_EXPECT(shared_);
    CELER_EXPECT(particle.particle_id() == shared_.mu_minus_id
                 || particle.particle_id() == shared_.mu_plus_id);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the muon decay.
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
            electron_nu_energy_frac = generate_canonical(rng);
        } while (RejectionSampler(
            electron_nu_energy_frac * (real_type{1} - electron_nu_energy_frac),
            real_type{0.25})(rng));

        electron_energy_frac = generate_canonical(rng);
    } while (electron_nu_energy_frac + electron_energy_frac < real_type{1});

    // Decay isotropically in rest frame and boost secondaries to the lab frame
    auto charged_lep_fv = this->to_lab_frame(
        IsotropicDistribution{}(rng),
        this->calc_momentum(electron_energy_frac, shared_.electron_mass),
        shared_.electron_mass);

    // Return charged lepton only
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 1};
    result.secondaries[0].particle_id = sec_id_;
    // Interaction stores kinetic energy; FourVector stores total energy
    result.secondaries[0].energy
        = Energy{charged_lep_fv.energy - shared_.electron_mass.value()};
    result.secondaries[0].direction = make_unit_vector(charged_lep_fv.mom);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Boost secondary to the lab frame.
 *
 * \note This assumes the primary to be at rest and, thus, there is no need
 * to perform an inverse boost of the primary at the CM frame.
 */
CELER_FUNCTION FourVector MuDecayInteractor::to_lab_frame(Real3 const& dir,
                                                          Momentum momentum,
                                                          Mass mass) const
{
    CELER_EXPECT(is_soft_unit_vector(dir));
    CELER_EXPECT(momentum > zero_quantity());
    CELER_EXPECT(mass >= zero_quantity());

    auto lepton_fv = FourVector::from_mass_momentum(mass, momentum, dir);
    boost(boost_vector(inc_fourvec_), &lepton_fv);

    return lepton_fv;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate final particle momentum (or kinetic energy) from its sampled
 * fractional energy.
 */
CELER_FUNCTION auto
MuDecayInteractor::calc_momentum(real_type energy_frac,
                                 Mass mass) const -> Momentum
{
    return Momentum{std::sqrt(ipow<2>(energy_frac * max_energy_)
                              + 2 * energy_frac * max_energy_ * mass.value())};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

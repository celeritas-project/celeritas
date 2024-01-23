//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/SBEnergySampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"
#include "celeritas/em/distribution/SBEnergyDistHelper.hh"
#include "celeritas/em/distribution/SBEnergyDistribution.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"

#include "PhysicsConstants.hh"
#include "SBPositronXsCorrector.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy from the SeltzerBerger model.
 */
class SBEnergySampler
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using SBTable = NativeCRef<SeltzerBergerTableData>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION SBEnergySampler(SBTable const& differential_xs,
                                          Energy const& inc_energy,
                                          Energy const& gamma_cutoff,
                                          MaterialView const& material,
                                          ElementComponentId const& elcomp_id,
                                          Mass const& inc_mass,
                                          bool const is_electron);

    // Sample the bremsstrahlung photon energy with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// DATA ////
    // Differential cross section table
    SBTable const& differential_xs_;
    // Incident particle energy
    Energy const inc_energy_;
    // Production cutoff for gammas
    Energy const gamma_cutoff_;
    // Material in which interaction occurs
    MaterialView const& material_;
    // Element in which interaction occurs
    ElementComponentId const elcomp_id_;
    // Incident particle mass
    Mass const inc_mass_;
    // Incident particle identification flag
    bool const inc_particle_is_electron_;
    // Density correction
    real_type density_correction_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 */
CELER_FUNCTION
SBEnergySampler::SBEnergySampler(SBTable const& differential_xs,
                                 Energy const& inc_energy,
                                 Energy const& gamma_cutoff,
                                 MaterialView const& material,
                                 ElementComponentId const& elcomp_id,
                                 Mass const& inc_mass,
                                 bool const is_electron)
    : differential_xs_(differential_xs)
    , inc_energy_(inc_energy)
    , gamma_cutoff_(gamma_cutoff)
    , material_(material)
    , elcomp_id_(elcomp_id)
    , inc_mass_(inc_mass)
    , inc_particle_is_electron_(is_electron)
{
    // Density correction
    real_type density_factor = material.electron_density() * migdal_constant();
    real_type total_energy_val = inc_energy_.value() + inc_mass_.value();
    density_correction_ = density_factor * ipow<2>(total_energy_val);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the exiting energy by doing a table lookup and rejection.
 */
template<class Engine>
CELER_FUNCTION auto SBEnergySampler::operator()(Engine& rng) -> Energy
{
    // Outgoing photon secondary energy sampler
    Energy gamma_exit_energy;

    // Helper class preprocesses cross section bounds and calculates
    // distribution
    SBEnergyDistHelper sb_helper(
        differential_xs_,
        inc_energy_,
        material_.element_id(elcomp_id_),
        SBEnergyDistHelper::EnergySq{density_correction_},
        gamma_cutoff_);

    if (inc_particle_is_electron_)
    {
        // Rejection sample without modifying cross section
        SBEnergyDistribution<SBElectronXsCorrector> sample_gamma_energy(
            sb_helper, {});
        gamma_exit_energy = sample_gamma_energy(rng);
    }
    else
    {
        SBEnergyDistribution<SBPositronXsCorrector> sample_gamma_energy(
            sb_helper,
            {inc_mass_,
             material_.make_element_view(elcomp_id_),
             gamma_cutoff_,
             inc_energy_});
        gamma_exit_energy = sample_gamma_energy(rng);
    }

    return gamma_exit_energy;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

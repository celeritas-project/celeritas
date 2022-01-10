//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBEnergySampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Units.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleTrackView.hh"

#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/Types.hh"

#include "SeltzerBergerData.hh"

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
    //! Type aliases
    using Energy = units::MevEnergy;
    using Mass   = units::MevMass;
    using SBTable
        = SeltzerBergerTableData<Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION SBEnergySampler(const SBTable& differential_xs,
                                          const ParticleTrackView& particle,
                                          const Energy&       gamma_cutoff,
                                          const MaterialView& material,
                                          const ElementComponentId& elcomp_id,
                                          const Mass&               inc_mass,
                                          const bool is_electron);

    // Sample the bremsstrahlung photon energy with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// DATA ////
    // Differential cross section table
    const SBTable& differential_xs_;
    // Incident particle energy
    const Energy inc_energy_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Material in which interaction occurs
    const MaterialView& material_;
    // Element in which interaction occurs
    const ElementComponentId elcomp_id_;
    // Incident particle mass
    const Mass inc_mass_;
    // Incident particle identification flag
    const bool inc_particle_is_electron_;
    // Density correction
    real_type density_correction_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SBEnergySampler.i.hh"

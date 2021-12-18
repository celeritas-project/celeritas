//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RBEnergySampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Units.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/Types.hh"

#include "random/distributions/ReciprocalDistribution.hh"

#include "RelativisticBremData.hh"
#include "RelativisticBremDXsection.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy from the relativistic model.
 *
 */
class RBEnergySampler
{
  public:
    //!@{
    //! Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    RBEnergySampler(const RelativisticBremNativeRef& shared,
                    const ParticleTrackView&         particle,
                    const CutoffView&                cutoffs,
                    const MaterialView&              material,
                    const ElementComponentId&        elcomp_id);

    // Sample the bremsstrahlung photon energy with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    using ReciprocalSampler = ReciprocalDistribution<real_type>;

    //// DATA ////

    // Incident particle energy
    const real_type inc_energy_;
    // Production cutoff for gammas
    const real_type gamma_cutoff_;
    // Differential cross section calcuator
    RelativisticBremDXsection dxsec_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "RBEnergySampler.i.hh"

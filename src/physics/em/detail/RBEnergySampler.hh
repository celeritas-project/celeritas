//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
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

#include "RelativisticBremData.hh"
#include "RBDiffXsCalculator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the bremsstrahlung photon energy from the relativistic model.
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
    //// DATA ////

    // Square of minimum of incident particle energy and cutoff
    real_type tmin_sq_;
    // Square of production cutoff for gammas
    real_type tmax_sq_;
    // Differential cross section calcuator
    RBDiffXsCalculator calc_dxsec_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "RBEnergySampler.i.hh"

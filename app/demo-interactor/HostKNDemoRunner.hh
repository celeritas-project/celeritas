//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/HostKNDemoRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"
#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "KNDemoIO.hh"
#include "XsGridParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Run interactions on the host CPU.
 *
 * This is an analog to the celeritas::app::KNDemoRunner for device simulation
 * but does all the transport directly on the CPU side.
 */
class HostKNDemoRunner
{
  public:
    //!@{
    //! \name Type aliases

    using result_type = celeritas::app::KNDemoResult;
    using constSPParticleParams = std::shared_ptr<ParticleParams const>;
    using constSPXsGridParams = std::shared_ptr<XsGridParams const>;
    //!@}

  public:
    // Construct with parameters
    HostKNDemoRunner(constSPParticleParams particles, constSPXsGridParams xs);

    // Run given number of particles
    result_type operator()(celeritas::app::KNDemoRunArgs args);

  private:
    constSPParticleParams pparams_;
    constSPXsGridParams xsparams_;
    KleinNishinaData kn_data_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

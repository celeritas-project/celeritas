//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "KNDemoIO.hh"
#include "KNDemoKernel.hh"
#include "XsGridParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Run a bunch of interactions.
 *
 * \code
    KNDemoRunner run_demo(pparams, args);
    auto diagnosticss = run_demo(1e6, 10);
   \endcode
 */
class KNDemoRunner
{
  public:
    //!@{
    //! \name Type aliases

    using result_type = KNDemoResult;
    using constSPParticleParams = std::shared_ptr<ParticleParams const>;
    using constSPXsGridParams = std::shared_ptr<XsGridParams const>;
    //!@}

  public:
    // Construct with parameters
    KNDemoRunner(constSPParticleParams particles,
                 constSPXsGridParams xs,
                 DeviceGridParams solver);

    // Run with a given particle vector size and max iterations
    result_type operator()(KNDemoRunArgs args);

  private:
    constSPParticleParams pparams_;
    constSPXsGridParams xsparams_;
    DeviceGridParams launch_params_;
    KleinNishinaData kn_data_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

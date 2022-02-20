//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/ParticleParams.hh"
#include "physics/em/detail/KleinNishinaData.hh"
#include "KNDemoKernel.hh"
#include "KNDemoIO.hh"
#include "XsGridParams.hh"

namespace demo_interactor
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
    //! Type aliases
    using size_type   = celeritas::size_type;
    using result_type = KNDemoResult;
    using constSPParticleParams
        = std::shared_ptr<const celeritas::ParticleParams>;
    using constSPXsGridParams = std::shared_ptr<const XsGridParams>;
    //!@}

  public:
    // Construct with parameters
    KNDemoRunner(constSPParticleParams particles,
                 constSPXsGridParams   xs,
                 DeviceGridParams      solver);

    // Run with a given particle vector size and max iterations
    result_type operator()(KNDemoRunArgs args);

  private:
    constSPParticleParams                   pparams_;
    constSPXsGridParams                     xsparams_;
    DeviceGridParams                        launch_params_;
    celeritas::detail::KleinNishinaData     kn_data_;
};

//---------------------------------------------------------------------------//
} // namespace demo_interactor

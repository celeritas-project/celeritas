//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParamsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/ParticleParams.hh"
#include "RootLoader.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load ParticleParams by reading the imported particle data from the ROOT
 * file produced by the app/geant-exporter.
 *
 * \code
 *  ParticleParamsLoader particle_loader(root_loader);
 *  const auto particle_params = particle_loader();
 * \endcode
 *
 * \sa RootLoader
 */
class ParticleParamsLoader
{
  public:
    // Construct with RootLoader
    ParticleParamsLoader(RootLoader& root_loader);

    // Return constructed ParticleParams
    const std::shared_ptr<const ParticleParams> operator()();

  private:
    RootLoader root_loader_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

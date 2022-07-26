//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ParticleParams.hh"
#include "Primary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Primary generator construction arguments.
 */
struct PrimaryGeneratorOptions
{
    int       pdg;      //!< Primary PDG number
    real_type energy;   //!< [MeV]
    Real3     position; //!< [cm]
    Real3     direction;
    size_type num_events;
    size_type primaries_per_event;

    //! Whether the options are valid
    explicit operator bool() const
    {
        return pdg != 0 && energy >= 0 && num_events > 0
               && primaries_per_event > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Generate a vector of primaries.
 *
 * This simple helper class can be used to generate primary particles of a
 * single particle type with a fixed energy, position, and direction.
 */
class PrimaryGenerator
{
  public:
    //!@{
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using VecPrimary       = std::vector<Primary>;
    //!@}

  public:
    // Construct with options and shared particle data
    PrimaryGenerator(SPConstParticles, PrimaryGeneratorOptions);

    // Generate primary particles
    VecPrimary operator()();

  private:
    SPConstParticles        particles_;
    PrimaryGeneratorOptions options_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

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
#include "PDGNumber.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Primary generator construction arguments.
 */
struct PrimaryGeneratorOptions
{
    PDGNumber        pdg;
    units::MevEnergy energy{0};
    Real3            position{0,0,0};
    Real3            direction{0,0,1};
    size_type        num_events{};
    size_type        primaries_per_event{};

    //! Whether the options are valid
    explicit operator bool() const
    {
        return pdg && energy >= zero_quantity() && num_events > 0
               && primaries_per_event > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Generate a vector of primaries.
 *
 * This simple helper class can be used to generate primary particles of a
 * single particle type with a fixed energy, position, and direction. Each \c
 * operator() call will return a single event until \c num_events events have
 * been generated.
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
    PrimaryGenerator(SPConstParticles, const PrimaryGeneratorOptions&);

    // Generate primary particles from a single event
    VecPrimary operator()();

  private:
    size_type num_events_;
    size_type primaries_per_event_;
    Primary   primary_;
    size_type event_count_{0};
};

//---------------------------------------------------------------------------//
} // namespace celeritas

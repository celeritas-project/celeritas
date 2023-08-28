//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <random>
#include <vector>

#include "celeritas/Types.hh"

#include "PDGNumber.hh"
#include "ParticleParams.hh"
#include "Primary.hh"
#include "PrimaryGeneratorOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate a vector of primaries.
 *
 * This simple helper class can be used to generate primary particles of one or
 * more particle types with the energy, position, and direction sampled from
 * distributions. If more than one PDG number is specified, an equal number of
 * each particle type will be produced. Each \c operator() call will return a
 * single event until \c num_events events have been generated.
 *
 * \todo Hardcode engine to mt19937 and inherit from EventReaderInterface (or
 * even change that name).
 */
class PrimaryGenerator
{
  public:
    //!@{
    using EnergySampler = std::function<real_type(std::mt19937&)>;
    using PositionSampler = std::function<Real3(std::mt19937&)>;
    using DirectionSampler = std::function<Real3(std::mt19937&)>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using VecPrimary = std::vector<Primary>;
    //!@}

    struct Input
    {
        std::vector<PDGNumber> pdg;
        size_type num_events{};
        size_type primaries_per_event{};
        EnergySampler sample_energy;
        PositionSampler sample_pos;
        DirectionSampler sample_dir;
    };

  public:
    // Construct from user input
    static PrimaryGenerator
    from_options(SPConstParticles, PrimaryGeneratorOptions const&);

    // Construct with options and shared particle data
    PrimaryGenerator(SPConstParticles, Input const&);

    // Generate primary particles from a single event
    VecPrimary operator()(std::mt19937& rng);

  private:
    size_type num_events_{};
    size_type primaries_per_event_{};
    EnergySampler sample_energy_;
    PositionSampler sample_pos_;
    DirectionSampler sample_dir_;
    std::vector<ParticleId> particle_id_;
    size_type primary_count_{0};
    size_type event_count_{0};
};
//---------------------------------------------------------------------------//
}  // namespace celeritas

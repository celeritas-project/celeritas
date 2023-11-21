//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/io/EventIOInterface.hh"

#include "PDGNumber.hh"
#include "PrimaryGeneratorOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
struct Primary;

//---------------------------------------------------------------------------//
/*!
 * Generate a vector of primaries.
 *
 * This simple helper class can be used to generate primary particles of one or
 * more particle types with the energy, position, and direction sampled from
 * distributions. If more than one PDG number is specified, an equal number of
 * each particle type will be produced. Each \c operator() call will return a
 * single event until \c num_events events have been generated.
 */
class PrimaryGenerator : public EventReaderInterface
{
  public:
    //!@{
    //! \name Type aliases
    using EnergySampler = std::function<real_type(PrimaryGeneratorEngine&)>;
    using PositionSampler = std::function<Real3(PrimaryGeneratorEngine&)>;
    using DirectionSampler = std::function<Real3(PrimaryGeneratorEngine&)>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using result_type = std::vector<Primary>;
    //!@}

    struct Input
    {
        unsigned int seed{};
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

    //! Prevent copying and moving
    CELER_DELETE_COPY_MOVE(PrimaryGenerator);

    // Generate primary particles from a single event
    result_type operator()() final;

  private:
    size_type num_events_{};
    size_type primaries_per_event_{};
    EnergySampler sample_energy_;
    PositionSampler sample_pos_;
    DirectionSampler sample_dir_;
    std::vector<ParticleId> particle_id_;
    size_type primary_count_{0};
    size_type event_count_{0};
    PrimaryGeneratorEngine rng_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas

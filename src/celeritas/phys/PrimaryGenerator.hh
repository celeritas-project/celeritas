//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <random>
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
namespace inp
{
struct PrimaryGenerator;
}

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
 * \todo Refactor generators so that event ID is an input and vector of
 * primaries (which won't have an event ID) is output.
 */
class PrimaryGenerator : public EventReaderInterface
{
  public:
    //!@{
    //! \name Type aliases
    using PrimaryGeneratorEngine = std::mt19937;
    using EnergySampler = std::function<real_type(PrimaryGeneratorEngine&)>;
    using PositionSampler = std::function<Real3(PrimaryGeneratorEngine&)>;
    using DirectionSampler = std::function<Real3(PrimaryGeneratorEngine&)>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using result_type = std::vector<Primary>;
    using Input = inp::CorePrimaryGenerator;
    //!@}

  public:
    // Construct from user input (deprecated)
    static PrimaryGenerator
    from_options(SPConstParticles, PrimaryGeneratorOptions const&);

    // Construct from shared particle data and new input
    PrimaryGenerator(Input const&, ParticleParams const& particles);

    // Construct from particle IDs and new input
    PrimaryGenerator(Input const&, std::vector<ParticleId> particle_ids);

    //! Prevent copying and moving
    CELER_DELETE_COPY_MOVE(PrimaryGenerator);
    ~PrimaryGenerator() override = default;

    // Generate primary particles from a single event
    result_type operator()() final;

    //! Get total number of events
    size_type num_events() const override { return num_events_; }

    // Reseed RNG for interaction with celer-g4
    void seed(UniqueEventId);

  private:
    size_type num_events_{};
    size_type primaries_per_event_{};
    unsigned int seed_{};
    EnergySampler sample_energy_;
    PositionSampler sample_pos_;
    DirectionSampler sample_dir_;
    std::vector<ParticleId> particle_id_;
    size_type event_count_{0};
    PrimaryGeneratorEngine rng_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas

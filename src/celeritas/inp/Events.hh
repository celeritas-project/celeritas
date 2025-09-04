//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Events.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <variant>

#include "corecel/Types.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//! Generate at a single point
struct PointDistribution
{
    Real3 pos{0, 0, 0};  // [length]
};

//! Sample uniformly in a box
struct UniformBoxDistribution
{
    Real3 lower{0, 0, 0};  // [length]
    Real3 upper{0, 0, 0};  // [length]
};

// TODO: cylinder shape
// TODO: shape with volume rejection

//! Choose a spatial distribution for the primary generator
using ShapeDistribution
    = std::variant<PointDistribution, UniformBoxDistribution>;

//---------------------------------------------------------------------------//
//! Generate angles isotropically
struct IsotropicDistribution
{
};

//! Generate angles in a single direction
struct MonodirectionalDistribution
{
    Real3 dir{0, 0, 1};
};

//! Choose an angular distribution for the primary generator
using AngleDistribution
    = std::variant<IsotropicDistribution, MonodirectionalDistribution>;

//---------------------------------------------------------------------------//
//! Generate primaries at a single energy value
struct MonoenergeticDistribution
{
    units::MevEnergy energy;
};

//! Choose an energy distribution for the primary generator
using EnergyDistribution = MonoenergeticDistribution;

//---------------------------------------------------------------------------//
/*!
 * Generate from a hardcoded distribution of primary particles.
 *
 * \todo move num_events to StandaloneInput
 */
struct PrimaryGenerator
{
    //! Number of events to generate
    size_type num_events{};
    //! Number of primaries per event
    size_type primaries_per_event{};

    //! Distribution for sampling spatial component (position)
    ShapeDistribution shape;
    //! Distribution for sampling angular component (direction)
    AngleDistribution angle;
    //! Distribution for sampling source energy
    EnergyDistribution energy;

    //! True if there's at least one primary
    explicit operator bool() const
    {
        return num_events > 0 && primaries_per_event > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Generate particles in the core stepping loop.
 *
 * \todo Allow programmatic setting from particle ID as well:
 * \code using Particle = std::variant<PDGNumber, ParticleId>; \endcode
 */
struct CorePrimaryGenerator : PrimaryGenerator
{
    //! Random number seed
    unsigned int seed{};
    //! Sample evenly from this vector of particle types
    std::vector<PDGNumber> pdg;

    //! True if there's at least one primary
    explicit operator bool() const
    {
        return PrimaryGenerator::operator bool() && !pdg.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Generate optical photon primary particles.
 *
 * \todo Time? Polarization?
 */
using OpticalPrimaryGenerator = PrimaryGenerator;

//---------------------------------------------------------------------------//
/*!
 * Sample random events from an input file.
 *
 * \todo move num_events to StandaloneInput
 */
struct SampleFileEvents
{
    //! Total number of events to sample
    size_type num_events{};
    //! File events per sampled event
    size_type num_merged{};

    //! ROOT file input
    std::string event_file;

    //! Random number generator seed
    unsigned int seed{};
};

//---------------------------------------------------------------------------//
//! Read all events from the given file
struct ReadFileEvents
{
    std::string event_file;
};

//---------------------------------------------------------------------------//
//! Mechanism for generating events for tracking
using Events
    = std::variant<CorePrimaryGenerator, SampleFileEvents, ReadFileEvents>;

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

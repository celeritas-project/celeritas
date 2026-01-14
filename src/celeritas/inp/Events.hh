//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Events.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <variant>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/inp/Distributions.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//! Generate at a single energy value [MeV]
using MonoenergeticDistribution = DeltaDistribution<double>;

//! Choose an energy distribution for the primary generator
using EnergyDistribution
    = std::variant<MonoenergeticDistribution, NormalDistribution>;

//---------------------------------------------------------------------------//
//! Generate at a single point
using PointDistribution = DeltaDistribution<Array<double, 3>>;

// TODO: cylinder shape
// TODO: shape with volume rejection

//! Choose a spatial distribution for the primary generator
using ShapeDistribution
    = std::variant<PointDistribution, UniformBoxDistribution>;

//---------------------------------------------------------------------------//
//! Generate angles in a single direction
using MonodirectionalDistribution = DeltaDistribution<Array<double, 3>>;

//! Choose an angular distribution for the primary generator
using AngleDistribution
    = std::variant<MonodirectionalDistribution, IsotropicDistribution>;

//---------------------------------------------------------------------------//
/*!
 * Generate from a hardcoded distribution of primary particles.
 */
struct PrimaryGenerator
{
    //! Distribution for sampling spatial component (position)
    ShapeDistribution shape;
    //! Distribution for sampling angular component (direction)
    AngleDistribution angle;
    //! Distribution for sampling source energy
    EnergyDistribution energy;
};

//---------------------------------------------------------------------------//
/*!
 * Generate particles in the core stepping loop.
 *
 * \todo move num_events to StandaloneInput
 * \todo Allow programmatic setting from particle ID as well:
 * \code using Particle = std::variant<PDGNumber, ParticleId>; \endcode
 */
struct CorePrimaryGenerator : PrimaryGenerator
{
    //! Number of events to generate
    size_type num_events{};
    //! Number of primaries per event
    size_type primaries_per_event{};

    //! Random number seed
    unsigned int seed{};
    //! Sample evenly from this vector of particle types
    std::vector<PDGNumber> pdg;

    //! True if there's at least one primary
    explicit operator bool() const
    {
        return num_events > 0 && primaries_per_event > 0 && !pdg.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Generate optical photon primary particles.
 *
 * \note The sampled optical photon primaries are unpolarized.
 */
struct OpticalPrimaryGenerator : PrimaryGenerator
{
    //! Total number of primaries
    size_type primaries{};

    //! True if there's at least one primary
    explicit operator bool() const { return primaries > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons from EM particles in Celeritas.
 */
struct OpticalEmGenerator
{
};

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons from offloaded distribution data.
 */
struct OpticalOffloadGenerator
{
};

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons directly from optical track initializers.
 */
struct OpticalDirectGenerator
{
};

//---------------------------------------------------------------------------//
//! Mechanism for generating optical photons
using OpticalGenerator = std::variant<OpticalEmGenerator,
                                      OpticalOffloadGenerator,
                                      OpticalPrimaryGenerator,
                                      OpticalDirectGenerator>;

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

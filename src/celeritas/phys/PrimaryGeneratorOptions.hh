//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <functional>
#include <random>

#include "corecel/io/StringEnumMapper.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"

#include "PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Distribution selection for sampling quantities in a \c PrimaryGenerator
enum class DistributionSelection
{
    delta,
    isotropic,
    box,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Distribution type and parameters.
 */
struct DistributionOptions
{
    DistributionSelection distribution{DistributionSelection::size_};
    std::vector<real_type> params;

    //! Whether the options are valid
    explicit operator bool() const
    {
        return distribution != DistributionSelection::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Primary generator options.
 *
 * TODO: distributions should be std::variant (see ORANGE input)
 *
 * - \c seed: RNG seed
 * - \c pdg: PDG numbers of the primaries. An equal number of primaries of each
 *   type will be generated
 * - \c num_events: total number of events to generate
 * - \c primaries_per_event: number of primaries to generate in each event
 * - \c energy: energy distribution type and parameters
 * - \c position: spatial distribution type and parameters
 * - \c direction: angular distribution type and parameters
 */
struct PrimaryGeneratorOptions
{
    unsigned int seed{};
    std::vector<PDGNumber> pdg;
    size_type num_events{};
    size_type primaries_per_event{};
    DistributionOptions energy;
    DistributionOptions position;
    DistributionOptions direction;

    //! Whether the options are valid
    explicit operator bool() const
    {
        return !pdg.empty()
               && std::all_of(pdg.begin(), pdg.end(), LogicalTrue{})
               && num_events > 0 && primaries_per_event > 0 && energy
               && position && direction;
    }
};

// TODO: move to PrimaryGenerator.hh

using PrimaryGeneratorEngine = std::mt19937;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get a distribution name
char const* to_cstring(DistributionSelection value);

// TODO: move these to PrimaryGenerator.hh
//! \cond

// Return a distribution for sampling the energy
std::function<real_type(PrimaryGeneratorEngine&)>
make_energy_sampler(DistributionOptions options);

// Return a distribution for sampling the position
std::function<Real3(PrimaryGeneratorEngine&)>
make_position_sampler(DistributionOptions options);

// Return a distribution for sampling the direction
std::function<Real3(PrimaryGeneratorEngine&)>
make_direction_sampler(DistributionOptions options);

//! \endcond
//---------------------------------------------------------------------------//
}  // namespace celeritas

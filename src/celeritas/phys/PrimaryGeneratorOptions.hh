//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptions.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/io/StringEnumMapper.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"

#include "PDGNumber.hh"

namespace celeritas
{
namespace inp
{
struct CorePrimaryGenerator;
}

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
 * - \c seed: RNG seed
 * - \c pdg: PDG numbers of the primaries. An equal number of primaries of each
 *   type will be generated
 * - \c num_events: total number of events to generate
 * - \c primaries_per_event: number of primaries to generate in each event
 * - \c energy: energy distribution type and parameters
 * - \c position: spatial distribution type and parameters
 * - \c direction: angular distribution type and parameters
 *
 * \deprecated See inp::PrimaryGenerator
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
        return !pdg.empty() && std::all_of(pdg.begin(), pdg.end(), Identity{})
               && num_events > 0 && primaries_per_event > 0 && energy
               && position && direction;
    }
};

//---------------------------------------------------------------------------//
// Convert PrimaryGeneratorOptions to inp::CorePrimaryGenerator.
inp::CorePrimaryGenerator to_input(PrimaryGeneratorOptions const&);

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get a distribution name
char const* to_cstring(DistributionSelection value);

//---------------------------------------------------------------------------//
}  // namespace celeritas

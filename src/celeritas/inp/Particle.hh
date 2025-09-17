//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Particle.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Types.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Branching ratio and daughters for a decay channel.
 */
struct DecayChannel
{
    DecayChannelType type{DecayChannelType::size_};
    double branching_ratio{};
    std::vector<PDGNumber> daughters;
};

//---------------------------------------------------------------------------//
/*!
 * Particle data.
 */
struct Particle
{
    using Charge = Quantity<units::EElectron, double>;
    using MevMass = Quantity<units::MevPerCsq, double>;
    using DecayTable = std::vector<DecayChannel>;

    //! Particle name
    std::string name;
    //! PDG code (see "Review of Particle Physics")
    PDGNumber pdg;
    //! Rest mass [MeV / c^2]
    MevMass mass;
    //! Elementary charge [e]
    Charge charge;
    //! Decay constant [1/time]
    double decay_constant{constants::stable_decay_constant};
    //! Decay channels
    DecayTable decay_channels;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

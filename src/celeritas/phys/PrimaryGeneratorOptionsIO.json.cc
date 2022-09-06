//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorOptionsIO.json.hh"

#include "corecel/cont/Array.json.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(const nlohmann::json& j, PrimaryGeneratorOptions& opts)
{
    int pdg;
    j.at("pdg").get_to(pdg);
    opts.pdg = PDGNumber{pdg};
    CELER_VALIDATE(opts.pdg, << "invalid PDG number " << pdg);
    // TODO: allow symmetric reading with JSON output (skipping "MeV")
    real_type energy;
    j.at("energy").get_to(energy);
    opts.energy = units::MevEnergy{energy};

    j.at("position").get_to(opts.position);
    j.at("direction").get_to(opts.direction);
    j.at("num_events").get_to(opts.num_events);
    j.at("primaries_per_event").get_to(opts.primaries_per_event);
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, const PrimaryGeneratorOptions& opts)
{
    j = nlohmann::json{{"pdg", opts.pdg.unchecked_get()},
                       {"energy", {opts.energy.value(), "MeV"}},
                       {"position", opts.position},
                       {"direction", opts.direction},
                       {"num_events", opts.num_events},
                       {"primaries_per_event", opts.primaries_per_event}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

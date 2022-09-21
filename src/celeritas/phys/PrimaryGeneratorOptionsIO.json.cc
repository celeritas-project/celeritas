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
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the distribution type.
 */
const char* to_cstring(DistributionSelection value)
{
    CELER_EXPECT(value != DistributionSelection::size_);

    static const char* const strings[] = {
        "delta",
        "isotropic",
        "box",
    };
    static_assert(
        static_cast<int>(DistributionSelection::size_) * sizeof(const char*)
            == sizeof(strings),
        "Enum strings are incorrect");

    return strings[static_cast<int>(value)];
}

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// JSON serializers
//---------------------------------------------------------------------------//
void from_json(const nlohmann::json& j, DistributionSelection& value)
{
    static auto from_string
        = StringEnumMap<DistributionSelection>::from_cstring_func(
            to_cstring, "distribution type");
    value = from_string(j.get<std::string>());
}

void to_json(nlohmann::json& j, const DistributionSelection& value)
{
    j = std::string{to_cstring(value)};
}

void from_json(const nlohmann::json& j, DistributionOptions& opts)
{
    j.at("distribution").get_to(opts.distribution);
    j.at("params").get_to(opts.params);
}

void to_json(nlohmann::json& j, const DistributionOptions& opts)
{
    j = nlohmann::json{{"distribution", opts.distribution},
                       {"params", opts.params}};
}

//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(const nlohmann::json& j, PrimaryGeneratorOptions& opts)
{
    std::vector<int> pdg;
    j.at("pdg").get_to(pdg);
    opts.pdg.reserve(pdg.size());
    for (int i : pdg)
    {
        PDGNumber p{i};
        opts.pdg.push_back(p);
        CELER_VALIDATE(p, << "invalid PDG number " << i);
    }
    j.at("num_events").get_to(opts.num_events);
    j.at("primaries_per_event").get_to(opts.primaries_per_event);
    j.at("energy").get_to(opts.energy);
    j.at("position").get_to(opts.position);
    j.at("direction").get_to(opts.direction);
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, const PrimaryGeneratorOptions& opts)
{
    std::vector<int> pdg(opts.pdg.size());
    std::transform(
        opts.pdg.begin(), opts.pdg.end(), pdg.begin(), [](PDGNumber p) {
            return p.unchecked_get();
        });
    j = nlohmann::json{{"pdg", pdg},
                       {"num_events", opts.num_events},
                       {"primaries_per_event", opts.primaries_per_event},
                       {"energy", opts.energy},
                       {"position", opts.position},
                       {"direction", opts.direction}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

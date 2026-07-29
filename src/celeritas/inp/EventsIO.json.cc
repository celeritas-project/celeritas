//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/EventsIO.json.cc
//---------------------------------------------------------------------------//
#include "EventsIO.json.hh"

#include <variant>
#include <vector>

#include "corecel/inp/DistributionsIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, EnergyDistribution const& v)
{
    j = std::visit([](auto const& d) { return nlohmann::json(d); }, v);
}

void from_json(nlohmann::json const& j, EnergyDistribution& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, delta, MonoenergeticDistribution);
    CELER_JSON_LOAD_VARIANT(j, v, normal, NormalDistribution);
    CELER_JSON_LOAD_VARIANT(
        j, v, truncated, TruncatedDistribution<NormalDistribution>);
    CELER_VALIDATE(false, << "invalid EnergyDistribution input");
}

void to_json(nlohmann::json& j, ShapeDistribution const& v)
{
    j = std::visit([](auto const& d) { return nlohmann::json(d); }, v);
}

void from_json(nlohmann::json const& j, ShapeDistribution& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, delta, PointDistribution);
    CELER_JSON_LOAD_VARIANT(j, v, uniform_box, UniformBoxDistribution);
    CELER_VALIDATE(false, << "invalid ShapeDistribution input");
}

void to_json(nlohmann::json& j, AngleDistribution const& v)
{
    j = std::visit([](auto const& d) { return nlohmann::json(d); }, v);
}

void from_json(nlohmann::json const& j, AngleDistribution& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, delta, MonodirectionalDistribution);
    CELER_JSON_LOAD_VARIANT(j, v, isotropic, IsotropicDistribution);
    CELER_VALIDATE(false, << "invalid AngleDistribution input");
}

void to_json(nlohmann::json& j, OpticalPrimaryGenerator const& v)
{
    j = nlohmann::json{
        json_type_pair("primary"),
        CELER_JSON_PAIR(v, shape),
        CELER_JSON_PAIR(v, angle),
        CELER_JSON_PAIR(v, energy),
        CELER_JSON_PAIR(v, primaries),
    };
}

void from_json(nlohmann::json const& j, OpticalPrimaryGenerator& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, shape);
    CELER_JSON_LOAD_REQUIRED(j, v, angle);
    CELER_JSON_LOAD_REQUIRED(j, v, energy);
    CELER_JSON_LOAD_REQUIRED(j, v, primaries);
}

void to_json(nlohmann::json& j, OpticalEmGenerator const&)
{
    j = nlohmann::json{json_type_pair("em")};
}

void from_json(nlohmann::json const&, OpticalEmGenerator&) {}

void to_json(nlohmann::json& j, OpticalOffloadGenerator const& v)
{
    j = nlohmann::json{
        json_type_pair("offload"),
        CELER_JSON_PAIR_OPTIONAL(v, distribution_file),
    };
}

void from_json(nlohmann::json const& j, OpticalOffloadGenerator& v)
{
    CELER_JSON_LOAD_OPTIONAL(j, v, distribution_file);
}

void to_json(nlohmann::json& j, OpticalDirectGenerator const&)
{
    j = nlohmann::json{json_type_pair("direct")};
}

void from_json(nlohmann::json const&, OpticalDirectGenerator&) {}

void to_json(nlohmann::json& j, OpticalGenerator const& v)
{
    j = std::visit([](auto const& g) { return nlohmann::json(g); }, v);
}

void from_json(nlohmann::json const& j, OpticalGenerator& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, primary, OpticalPrimaryGenerator);
    CELER_JSON_LOAD_VARIANT(j, v, em, OpticalEmGenerator);
    CELER_JSON_LOAD_VARIANT(j, v, offload, OpticalOffloadGenerator);
    CELER_JSON_LOAD_VARIANT(j, v, direct, OpticalDirectGenerator);
    CELER_VALIDATE(false, << "invalid OpticalGenerator input");
}

void to_json(nlohmann::json& j, CorePrimaryGenerator const& v)
{
    std::vector<int> pdg(v.pdg.size());
    std::transform(v.pdg.begin(), v.pdg.end(), pdg.begin(), [](PDGNumber p) {
        return p.unchecked_get();
    });

    j = nlohmann::json{
        json_type_pair("primary"),
        CELER_JSON_PAIR(v, shape),
        CELER_JSON_PAIR(v, angle),
        CELER_JSON_PAIR(v, energy),
        CELER_JSON_PAIR(v, num_events),
        CELER_JSON_PAIR(v, primaries_per_event),
        CELER_JSON_PAIR(v, seed),
        {"pdg", pdg},
    };
}

void from_json(nlohmann::json const& j, CorePrimaryGenerator& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, shape);
    CELER_JSON_LOAD_REQUIRED(j, v, angle);
    CELER_JSON_LOAD_REQUIRED(j, v, energy);
    CELER_JSON_LOAD_REQUIRED(j, v, num_events);
    CELER_JSON_LOAD_REQUIRED(j, v, primaries_per_event);
    CELER_JSON_LOAD_OPTION(j, v, seed);

    std::vector<int> pdg;
    j.at("pdg").get_to(pdg);
    v.pdg.reserve(pdg.size());
    for (int i : pdg)
    {
        PDGNumber p{i};
        v.pdg.push_back(p);
        CELER_VALIDATE(p, << "invalid PDG number " << i);
    }
}

void to_json(nlohmann::json& j, SampleFileEvents const& v)
{
    j = nlohmann::json{
        json_type_pair("sample"),
        CELER_JSON_PAIR(v, num_events),
        CELER_JSON_PAIR(v, num_merged),
        CELER_JSON_PAIR(v, event_file),
        CELER_JSON_PAIR(v, seed),
    };
}

void from_json(nlohmann::json const& j, SampleFileEvents& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, num_events);
    CELER_JSON_LOAD_REQUIRED(j, v, num_merged);
    CELER_JSON_LOAD_REQUIRED(j, v, event_file);
    CELER_JSON_LOAD_REQUIRED(j, v, seed);
}

void to_json(nlohmann::json& j, ReadFileEvents const& v)
{
    j = nlohmann::json{
        json_type_pair("read"),
        CELER_JSON_PAIR(v, event_file),
    };
}

void from_json(nlohmann::json const& j, ReadFileEvents& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, event_file);
}

void to_json(nlohmann::json& j, Generator const& v)
{
    j = std::visit([](auto const& g) { return nlohmann::json(g); }, v);
}

void from_json(nlohmann::json const& j, Generator& v)
{
    CELER_JSON_LOAD_VARIANT(j, v, primary, CorePrimaryGenerator);
    CELER_JSON_LOAD_VARIANT(j, v, sample, SampleFileEvents);
    CELER_JSON_LOAD_VARIANT(j, v, read, ReadFileEvents);
    CELER_VALIDATE(false, << "invalid Generator input");
}

void to_json(nlohmann::json& j, Events const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, generator),
        CELER_JSON_PAIR(v, merge),
    };
}

void from_json(nlohmann::json const& j, Events& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, generator);
    CELER_JSON_LOAD_OPTION(j, v, merge);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

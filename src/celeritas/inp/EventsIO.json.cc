//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/EventsIO.json.cc
//---------------------------------------------------------------------------//
#include "EventsIO.json.hh"

#include <variant>

#include "corecel/inp/DistributionsIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

#define EIO_LOAD_VARIANT(NAME, TYPE) \
    do                               \
    {                                \
        if (j.at("_type") == #NAME)  \
        {                            \
            v = j.get<TYPE>();       \
            return;                  \
        }                            \
    } while (0)

void to_json(nlohmann::json& j, EnergyDistribution const& v)
{
    j = std::visit([](auto const& d) { return nlohmann::json(d); }, v);
}

void from_json(nlohmann::json const& j, EnergyDistribution& v)
{
    EIO_LOAD_VARIANT(delta, MonoenergeticDistribution);
    EIO_LOAD_VARIANT(normal, NormalDistribution);
    CELER_VALIDATE(false, << "invalid EnergyDistribution input");
}

void to_json(nlohmann::json& j, ShapeDistribution const& v)
{
    j = std::visit([](auto const& d) { return nlohmann::json(d); }, v);
}

void from_json(nlohmann::json const& j, ShapeDistribution& v)
{
    EIO_LOAD_VARIANT(delta, PointDistribution);
    EIO_LOAD_VARIANT(uniform_box, UniformBoxDistribution);
    CELER_VALIDATE(false, << "invalid ShapeDistribution input");
}

void to_json(nlohmann::json& j, AngleDistribution const& v)
{
    j = std::visit([](auto const& d) { return nlohmann::json(d); }, v);
}

void from_json(nlohmann::json const& j, AngleDistribution& v)
{
    EIO_LOAD_VARIANT(delta, MonodirectionalDistribution);
    EIO_LOAD_VARIANT(isotropic, IsotropicDistribution);
    CELER_VALIDATE(false, << "invalid AngleDistribution input");
}

void to_json(nlohmann::json& j, OpticalPrimaryGenerator const& v)
{
    j = nlohmann::json{
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

#undef EIO_LOAD_VARIANT

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

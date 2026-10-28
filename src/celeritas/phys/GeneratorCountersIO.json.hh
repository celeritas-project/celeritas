//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/GeneratorCountersIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"

#include "GeneratorCounters.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template<class T>
void to_json(nlohmann::json& j, GeneratorCounters<T> const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, buffer_size),
        CELER_JSON_PAIR(v, num_pending),
        CELER_JSON_PAIR(v, num_generated),
    };
}

template<class T>
void from_json(nlohmann::json const& j, GeneratorCounters<T>& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, buffer_size);
    CELER_JSON_LOAD_REQUIRED(j, v, num_pending);
    CELER_JSON_LOAD_REQUIRED(j, v, num_generated);
}

void to_json(nlohmann::json& j, CounterAccumStats const&);
void from_json(nlohmann::json const& j, CounterAccumStats&);

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file JsonUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Verify JSON round-trip serialization.
template<class T>
inline void verify_json_round_trip(T const& input, char const* expected)
{
    nlohmann::json obj(input);
    EXPECT_JSON_EQ(expected, obj.dump());

    auto rt_input = obj.get<T>();
    EXPECT_JSON_EQ(expected, nlohmann::json(rt_input).dump());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

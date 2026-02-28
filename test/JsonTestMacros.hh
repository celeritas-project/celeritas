//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file JsonTestMacros.hh
//! \brief Test assertions that required nlohmann_json to be linked in
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "testdetail/TestMacrosImpl.hh"

#define EXPECT_JSON_ROUND_TRIP(OBJ, EXPECTED) \
    EXPECT_PRED_FORMAT2(::celeritas::testdetail::JsonRoundTrip, OBJ, EXPECTED)

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
//! Verify JSON round-trip serialization.
template<class T>
inline ::testing::AssertionResult JsonRoundTrip(char const* input_expr,
                                                char const*,
                                                T const& input,
                                                std::string_view expected)
{
    // Check serialization
    nlohmann::json obj(input);
    ::testing::AssertionResult result = IsJsonEq("", "", expected, obj.dump());

    if (!result)
    {
        result << "\nSerialization of " << input_expr << " failed";
        return result;
    }

    // Check deserialization since serialization worked
    auto rt_input = obj.get<T>();
    // Verify equality by reserializing it rather than operator==
    result = IsJsonEq("", "", expected, nlohmann::json(rt_input).dump());
    if (!result)
    {
        result << "\nDeserialization of " << input_expr << " failed";
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas

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
inline ::testing::AssertionResult JsonRoundTrip(char const* expr1,
                                                char const* expr2,
                                                T const& input,
                                                std::string_view expected)
{
    // Check serialization
    nlohmann::json obj(input);
    std::string actual_expr{"json("};
    actual_expr += expr2;
    actual_expr += ").dump()";
    ::testing::AssertionResult result
        = IsJsonEq(expr1, actual_expr.c_str(), expected, obj.dump());

    if (result)
    {
        // Check deserialization since serialization worked
        auto rt_input = obj.get<T>();
        // Verify equality by reserializing it rather than operator==
        result = IsJsonEq(expr1,
                          "json(reconstructed).dump()",
                          expected,
                          nlohmann::json(rt_input).dump());
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/VariantUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/VariantUtils.hh"

#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "celeritas_test.hh"
// #include "VariantUtils.test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T>
std::string generic_to_string(T&& val)
{
    std::ostringstream os;
    os << val;
    return os.str();
}

TEST(ContainerVisitorTest, all)
{
    using VarT = std::variant<int, std::string>;
    using VecT = std::vector<VarT>;

    std::vector<std::string> result;
    auto append_to_result
        = [&result](auto&& v) { result.push_back(generic_to_string(v)); };

    ContainerVisitor visit{VecT{1, "three", 0, "two"}};
    EXPECT_TRUE((std::is_same_v<decltype(visit), ContainerVisitor<VecT>>));
    visit(append_to_result, 2);
    visit(append_to_result, 0);
    visit(append_to_result, 3);
    visit(append_to_result, 1);

    static char const* const expected_result[] = {"0", "1", "two", "three"};
    EXPECT_VEC_EQ(expected_result, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MagFieldEquation.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/MagFieldEquation.hh"

#include "celeritas_test.hh"

using celeritas::units::ElementaryCharge;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

Real3 dummy_field(Real3 const& pos)
{
    // Rotate and scale
    Real3 result;
    result[0] = real_type(0.5) * pos[1];
    result[1] = real_type(1.0) * pos[2];
    result[2] = real_type(2.0) * pos[0];
    return result;
}

template<class FieldT>
CELER_FUNCTION decltype(auto)
make_equation(FieldT&& field, ElementaryCharge charge)
{
    using Equation_t = celeritas::MagFieldEquation<FieldT>;
    return Equation_t{::celeritas::forward<FieldT>(field), charge};
}

void print_expected(OdeState const& s)
{
    cout << "/*** BEGIN CODE ***/\n"
         << "EXPECT_VEC_SOFT_EQ(Real3(" << repr(s.pos) << "), result.pos);\n"
         << "EXPECT_VEC_SOFT_EQ(Real3(" << repr(s.mom) << "), result.mom);\n"
         << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//

TEST(MagFieldEquationTest, charged)
{
    auto eval = make_equation(dummy_field, ElementaryCharge{3});
    {
        OdeState result = eval({{1, 2, 3}, {0, 0, 1}});
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), result.pos);
        EXPECT_VEC_SOFT_EQ(Real3({-0.002698132122, 0.000899377374, 0}),
                           result.mom);
    }
    {
        OdeState result = eval({{0.5, -2, -1}, {1, 2, 3}});
        EXPECT_VEC_SOFT_EQ(
            Real3({0.26726124191242, 0.53452248382485, 0.80178372573727}),
            result.pos);
        EXPECT_VEC_SOFT_EQ(Real3({0.0012018435696159,
                                  -0.0009614748556927,
                                  0.00024036871392318}),
                           result.mom);
    }
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(eval({{0.5, -2, -1}, {0, 0, 0}}), DebugError);
    }
}

TEST(MagFieldEquationTest, neutral)
{
    auto eval = make_equation(dummy_field, ElementaryCharge{0});
    {
        OdeState s{{0.5, -2, -1}, {0, 0, 1}};
        OdeState result = eval(s);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), result.pos);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 0}), result.mom);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

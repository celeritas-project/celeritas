//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MagFieldEquation.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/MagFieldEquation.hh"

#include "celeritas/UnitUtils.hh"

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
    result[0] = real_type(0.5) * to_cm(pos[1]) * units::gauss;
    result[1] = real_type(1.0) * to_cm(pos[2]) * units::gauss;
    result[2] = real_type(2.0) * to_cm(pos[0]) * units::gauss;
    return result;
}

template<class FieldT>
CELER_FUNCTION decltype(auto)
make_equation(FieldT&& field, ElementaryCharge charge)
{
    using Equation_t = celeritas::MagFieldEquation<FieldT>;
    return Equation_t{::celeritas::forward<FieldT>(field), charge};
}

OdeState make_state(Real3 const& pos, Real3 const& mom)
{
    OdeState result;
    result.pos = from_cm(pos);
    result.mom = mom;
    return result;
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
    constexpr auto inv_cm = 1 / units::centimeter;

    // NOTE: result.pos (dx / ds) is unitless,
    //  and  result.dir (dp / ds) has units of 1/length
    auto eval = make_equation(dummy_field, ElementaryCharge{3});
    {
        OdeState result = eval(make_state({1, 2, 3}, {0, 0, 1}));
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), result.pos);
        EXPECT_VEC_SOFT_EQ(
            Real3({-0.002698132122 * inv_cm, 0.000899377374 * inv_cm, 0}),
            result.mom);
    }
    {
        OdeState result = eval(make_state({0.5, -2, -1}, {1, 2, 3}));
        EXPECT_VEC_SOFT_EQ(
            Real3({0.26726124191242, 0.53452248382485, 0.80178372573727}),
            result.pos);
        EXPECT_VEC_SOFT_EQ(Real3({0.0012018435696159 * inv_cm,
                                  -0.0009614748556927 * inv_cm,
                                  0.00024036871392318 * inv_cm}),
                           result.mom);
    }
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(eval(make_state(Real3{0.5, -2, -1}, {0, 0, 0})),
                     DebugError);
    }
}

TEST(MagFieldEquationTest, neutral)
{
    auto eval = make_equation(dummy_field, ElementaryCharge{0});
    {
        OdeState result = eval(make_state({0.5, -2, -1}, {0, 0, 1}));
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), result.pos);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 0}), result.mom);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/FourVector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/FourVector.hh"

#include <cmath>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

class DummyParticleView
{
  public:
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;

    Momentum momentum() const
    {
        return Momentum{std::sqrt(ipow<2>(energy_) + 2 * mass_ * energy_)};
    }
    Energy total_energy() const { return Energy{mass_ + energy_}; }

    Energy energy() const { return Energy{energy_}; }
    Mass mass() const { return Mass{mass_}; }

  private:
    real_type mass_{0.1};
    real_type energy_{1000};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(FourVectorTest, basic)
{
    FourVector a{{1, 2, 3}, 1000};
    FourVector b{{4, 5, 6}, 2000};
    FourVector c = a + b;

    Real3 const expected_mom = {5, 7, 9};
    EXPECT_SOFT_EQ(a.energy + b.energy, c.energy);
    EXPECT_SOFT_EQ(3000, c.energy);
    EXPECT_EQ(a.mom + b.mom, c.mom);
    EXPECT_EQ(expected_mom, c.mom);
    EXPECT_DOUBLE_EQ(std::sqrt(ipow<2>(c.energy) - dot_product(c.mom, c.mom)),
                     norm(c));
    EXPECT_SOFT_EQ(2999.9741665554388, norm(c));

    FourVector d = a;
    d += b;
    EXPECT_SOFT_EQ(a.energy + b.energy, d.energy);
    EXPECT_SOFT_EQ(3000, d.energy);
    EXPECT_EQ(a.mom + b.mom, d.mom);
    EXPECT_EQ(expected_mom, d.mom);
}

TEST(FourVectorTest, boost_vector)
{
    FourVector a{{1, 2, 3}, 1000};
    Real3 expected_boost_vector = {0.001, 0.002, 0.003};
    EXPECT_VEC_SOFT_EQ(a.mom / a.energy, boost_vector(a));
    EXPECT_VEC_SOFT_EQ(expected_boost_vector, boost_vector(a));
}

TEST(FourVectorTest, boost)
{
    FourVector const a{{1, 2, 3}, 1000};
    FourVector b{{4, 5, 6}, 2000};
    boost(boost_vector(a), &b);

    EXPECT_VEC_SOFT_EQ(
        (Real3{6.0000300003150038, 9.0000600006300076, 12.000090000945011}),
        b.mom);
    EXPECT_SOFT_EQ(2000.0460003710041, b.energy);
}

TEST(FourVectorTest, constructors)
{
    DummyParticleView const p;

    auto fv = FourVector::from_mass_momentum(
        p.mass(), p.momentum(), Real3{0, 0, 1});
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1000.0999950005}), fv.mom);
    EXPECT_SOFT_EQ(1000.1, fv.energy);

    auto fv2 = FourVector::from_particle(p, Real3{0, 0, 1});
    EXPECT_VEC_SOFT_EQ(fv.mom, fv2.mom);
    EXPECT_SOFT_EQ(fv.energy, fv2.energy);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

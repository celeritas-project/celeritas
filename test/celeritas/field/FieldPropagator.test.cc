//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/FieldPropagator.hh"

#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/field/detail/CMSParameterizedField.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "celeritas_test.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::constants::pi;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldPropagatorTestBase : public GlobalGeoTestBase
{
  public:
    SPConstMaterial build_material() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstCutoff  build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics build_physics() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction  build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

    SPConstParticle build_particle() override
    {
        using namespace celeritas::units;
        namespace pdg = celeritas::pdg;

        // Create particle defs
        constexpr auto        stable = ParticleRecord::stable_decay_constant();
        ParticleParams::Input defs   = {
              {"electron",
               pdg::electron(),
               MevMass{0.5109989461},
               ElementaryCharge{-1},
               stable},
              {"positron",
               pdg::positron(),
               MevMass{0.5109989461},
               ElementaryCharge{1},
               stable},
              {"gamma", pdg::gamma(), zero_quantity(), zero_quantity(), stable}};
        return std::make_shared<ParticleParams>(std::move(defs));
    }

    void SetUp() override
    {
        geo_state_ = GeoStateStore(*this->geometry(), 1);
        par_state_ = ParStateStore(*this->particle(), 1);
    }

    ParticleTrackView init_particle(ParticleId id, MevEnergy energy)
    {
        CELER_EXPECT(id && energy > zero_quantity());
        ParticleTrackView view{
            this->particle()->host_ref(), par_state_.ref(), ThreadId{0}};
        view = {id, energy};
        return view;
    }

    GeoTrackView init_geo(const Real3& pos, Real3 dir)
    {
        normalize_direction(&dir);
        GeoTrackView view{
            this->geometry()->host_ref(), geo_state_.ref(), ThreadId{0}};
        view = {pos, dir};
        return view;
    }

  private:
    //// TYPE ALIASES ////
    template<template<Ownership, MemSpace> class T>
    using HostStateStore = CollectionStateStore<T, MemSpace::host>;
    using GeoStateStore  = HostStateStore<GeoStateData>;
    using ParStateStore  = HostStateStore<ParticleStateData>;

    //// DATA ////

    GeoStateStore geo_state_;
    ParStateStore par_state_;
};

class TwoBoxTest : public FieldPropagatorTestBase
{
    const char* geometry_basename() const override { return "two-boxes"; }
};

class LayersTest : public FieldPropagatorTestBase
{
    const char* geometry_basename() const override { return "field-test"; }
};

TEST_F(TwoBoxTest, interior_stepping)
{
    // Initialize particle state
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10.9181415106});

    // Calculate expected radius of curvature
    const real_type field_strength{1.0 * units::tesla};
    const real_type radius
        = native_value_from(particle.momentum())
          / (std::fabs(native_value_from(particle.charge())) * field_strength);
    EXPECT_SOFT_EQ(3.8085385437789383, radius);

    // Initialize position and direction so its curved track is centered about
    // the origin, moving counterclockwise from the right
    auto geo = this->init_geo({radius, 0, 0}, {0, 1, 0});
    EXPECT_EQ("inner", this->geometry()->id_to_label(geo.volume_id()));

    // Construct field (uniform along +Z)
    UniformField field({0, 0, field_strength});
    // Default driver options
    FieldDriverOptions driver_options;

    // Build propagator
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, &geo);

    // Test step that's smaller than driver's minimum (won't actually alter geo
    // state but should return the distance as if it were moved)
    Propagation result = propagate(1e-10);
    EXPECT_DOUBLE_EQ(1e-10, result.distance);
    EXPECT_FALSE(result.boundary);
    EXPECT_VEC_SOFT_EQ(Real3({radius, 0, 0}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.dir());

    // Test a short step
    result = propagate(1e-2);
    EXPECT_SOFT_EQ(1e-2, result.distance);
    EXPECT_VEC_SOFT_EQ(Real3({3.80852541539105, 0.0099999885096862, 0}),
                       geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({-0.00262567606832303, 0.999996552906651, 0}),
                       geo.dir());

    // Test the remaining quarter-turn divided into 20 steps
    {
        real_type step = 0.5 * pi * radius - 1e-2;
        for (auto i : range(25))
        {
            SCOPED_TRACE(i);
            result = propagate(step / 25);
            EXPECT_SOFT_EQ(step / 25, result.distance);
            EXPECT_FALSE(result.boundary)
                << "At " << geo.pos() << " along " << geo.dir();
        }
        EXPECT_SOFT_NEAR(0, distance(Real3({0, radius, 0}), geo.pos()), 1e-8);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
    }

    // Test a very long (next quarter-turn) step
    {
        SCOPED_TRACE("Quarter turn");
        result = propagate(0.5 * pi * radius);
        EXPECT_SOFT_EQ(0.5 * pi * radius, result.distance);
        EXPECT_LT(distance(Real3({-radius, 0, 0}), geo.pos()), 1e-6);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, -1, 0}), geo.dir()));
    }

    // Test a ridiculously long (half-turn) step to put us back at the start
    {
        SCOPED_TRACE("Half turn");
        result = propagate(pi * radius);
        EXPECT_SOFT_EQ(pi * radius, result.distance);
        EXPECT_LT(distance(Real3({radius, 0, 0}), geo.pos()), 1e-5);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, 1, 0}), geo.dir()));
    }
}

//---------------------------------------------------------------------------//

TEST_F(LayersTest, uniform)
{
#if 0
        FieldTestParams test
        // Input parameters of an electron in a uniform magnetic field
        test.nstates     = 128;
        test.nsteps      = 100;
        test.revolutions = 10;
        test.field_value = 1.0 * units::tesla;
        test.radius      = 3.8085386036;
        test.delta_z     = 6.7003310629;
        test.energy      = 10.9181415106;
        test.momentum_y  = 11.4177114158018;
        test.momentum_z  = 0.0;
        test.epsilon     = 1.0e-5;
#endif
}

//---------------------------------------------------------------------------//

TEST_F(LayersTest, cms_parameterized_field)
{
#if 0
    celeritas_test::detail::CMSParameterizedField calc_field;
#endif
}

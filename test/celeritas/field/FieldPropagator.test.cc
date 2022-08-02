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
#include "celeritas/Constants.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "celeritas_test.hh"
#include "detail/CMSParameterizedField.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::constants::pi;
using celeritas::constants::sqrt_three;
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

    GeoTrackView make_geo_view()
    {
        return {this->geometry()->host_ref(), geo_state_.ref(), ThreadId{0}};
    }

    GeoTrackView init_geo(const Real3& pos, Real3 dir)
    {
        normalize_direction(&dir);
        GeoTrackView view = this->make_geo_view();
        view              = {pos, dir};
        return view;
    }

    template<class Field>
    real_type calc_field_curvature(const ParticleTrackView& particle,
                                   const GeoTrackView&      geo,
                                   const Field&             calc_field) const
    {
        auto field_strength = norm(calc_field(geo.pos()));
        return native_value_from(particle.momentum())
               / (std::fabs(native_value_from(particle.charge()))
                  * field_strength);
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

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//

// Field strength is zero for z <= 0, linearly increasing for z > 0 so that at
// z=1 it has a value of "strength"
struct ReluZField
{
    real_type strength;

    Real3 operator()(const Real3& pos) const
    {
        return {0, 0, this->strength * celeritas::max<real_type>(0, pos[2])};
    }
};

//---------------------------------------------------------------------------//
// CONSTANTS
//---------------------------------------------------------------------------//

// Field value (native units) for 10 MeV electron/positron to have a radius of
// 1 cm
constexpr real_type unit_radius_field_strength{3501.9461121752274};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TwoBoxTest, electron_interior)
{
    // Initialize position and direction so its curved track is centered about
    // the origin, moving counterclockwise from the right
    const real_type radius{3.8085385437789383};
    auto            particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10.9181415106});
    auto          geo = this->init_geo({radius, 0, 0}, {0, 1, 0});
    UniformZField field(1.0 * units::tesla);

    // Check expected field curvature and geometry cell
    EXPECT_SOFT_EQ(radius, this->calc_field_curvature(particle, geo, field));
    EXPECT_EQ("inner", this->geometry()->id_to_label(geo.volume_id()));

    // Build propagator
    FieldDriverOptions driver_options;
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

TEST_F(TwoBoxTest, positron_interior)
{
    // Initialize position and direction so its curved track (radius 1) is
    // centered about the origin, moving *clockwise* from the right
    const real_type radius{1.0};
    auto            particle = this->init_particle(
        this->particle()->find(pdg::positron()), MevEnergy{10});
    auto          geo = this->init_geo({radius, 0, 0}, {0, -1, 0});
    UniformZField field(unit_radius_field_strength);

    // Check expected field curvature
    EXPECT_SOFT_EQ(radius, this->calc_field_curvature(particle, geo, field));

    // Build propagator
    FieldDriverOptions driver_options;
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, &geo);

    // Test a quarter turn
    Propagation result = propagate(0.5 * pi * radius);
    EXPECT_SOFT_EQ(0.5 * pi * radius, result.distance);
    EXPECT_SOFT_NEAR(0, distance(Real3({0, -radius, 0}), geo.pos()), 1e-5);
    EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
}

// Gamma in magnetic field should have a linear path
TEST_F(TwoBoxTest, gamma_interior)
{
    auto particle = this->init_particle(this->particle()->find(pdg::gamma()),
                                        MevEnergy{1});

    // Construct field (magnitude shouldn't matter)
    UniformZField      field(1234.5);
    FieldDriverOptions driver_options;

    // Propagate inside box
    {
        auto geo       = this->init_geo({0, 0, 0}, {0, 0, 1});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(3.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 3}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
    }
    // Move to boundary
    {
        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(2.0, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 5}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
    }
    // Cross boundary
    {
        auto geo = this->make_geo_view();
        EXPECT_EQ("inner", this->geometry()->id_to_label(geo.volume_id()));
        geo.cross_boundary();
        EXPECT_EQ("world", this->geometry()->id_to_label(geo.volume_id()));
        EXPECT_FALSE(geo.is_outside());
    }
    // Move in new region
    {
        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(5.0);
        EXPECT_SOFT_EQ(5.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 10}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
    }
}

// Electron will be tangent to the boundary at the top of its curved path.
TEST_F(TwoBoxTest, electron_tangent)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    UniformZField      field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    {
        SCOPED_TRACE("Nearly quarter turn close to boundary");

        auto geo       = this->init_geo({1, 4, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(0.49 * pi);

        EXPECT_SOFT_EQ(0.49 * pi, result.distance);
        EXPECT_LT(
            distance(Real3({std::cos(0.49 * pi), 4 + std::sin(0.49 * pi), 0}),
                     geo.pos()),
            1e-6);
    }
    {
        SCOPED_TRACE("Short step tangent to boundary");

        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(0.02 * pi);

        EXPECT_SOFT_EQ(0.02 * pi, result.distance);
        EXPECT_LT(
            distance(Real3({std::cos(0.51 * pi), 4 + std::sin(0.51 * pi), 0}),
                     geo.pos()),
            1e-6);
    }
}

// Electron crosses and reenters
TEST_F(TwoBoxTest, electron_cross)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    UniformZField      field(0.5 * unit_radius_field_strength);
    FieldDriverOptions driver_options;

    {
        auto geo = this->init_geo({2, 4, 0}, {0, 1, 0});
        EXPECT_SOFT_EQ(2.0, this->calc_field_curvature(particle, geo, field));
    }
    const real_type circ = 2.0 * 2 * pi;

    {
        SCOPED_TRACE("Exit (twelfth of a turn)");

        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(pi);

        EXPECT_SOFT_NEAR(1. / 12., result.distance / circ, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({sqrt_three, 5, 0}), geo.pos()), 1e-5);
        // Direction should be up left
        EXPECT_LT(distance(Real3({-0.5, sqrt_three / 2, 0}), geo.dir()), 1e-5);
    }
    {
        SCOPED_TRACE("Cross boundary");

        auto geo = this->make_geo_view();
        EXPECT_EQ("inner", this->geometry()->id_to_label(geo.volume_id()));
        geo.cross_boundary();
        EXPECT_EQ("world", this->geometry()->id_to_label(geo.volume_id()));
        EXPECT_FALSE(geo.is_outside());
    }
    {
        SCOPED_TRACE("Reenter (1/3 turn)");

        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        EXPECT_SOFT_NEAR(1. / 3., result.distance / circ, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-sqrt_three, 5, 0}), geo.pos()), 1e-5);
        // Direction should be down left
        EXPECT_LT(distance(Real3({-0.5, -sqrt_three / 2, 0}), geo.dir()), 1e-5);
    }
    {
        SCOPED_TRACE("Cross boundary");

        auto geo = this->make_geo_view();
        geo.cross_boundary();
        EXPECT_EQ("inner", this->geometry()->id_to_label(geo.volume_id()));
    }
    {
        SCOPED_TRACE("Return to start (2/3 turn)");

        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(7. / 12. * circ);

        EXPECT_SOFT_NEAR(7. / 12., result.distance / circ, 1e-5);
        EXPECT_FALSE(result.boundary);
        EXPECT_LT(distance(Real3({2, 4, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({0, 1, 0}), geo.dir()), 1e-5);
    }
}

// Electron barely crosses boundary
// TODO: fails to hit boundary: it just does a whole turn "interior" to the
// inner volume
TEST_F(TwoBoxTest, DISABLED_electron_tangent_cross)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    UniformZField      field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    // Circumference
    const real_type circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits boundary");

        auto geo       = this->init_geo({1, 4 + 1e-3, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        EXPECT_SOFT_EQ(circ / 4, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({0, 5, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({-1, 0, 0}), geo.dir()), 1e-5);
    }
}

TEST_F(TwoBoxTest, nonuniform_field)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    ReluZField         field{unit_radius_field_strength};
    FieldDriverOptions driver_options;

    this->init_geo({-2.0, 0, 0}, {0, 1, 1});

    std::vector<Real3>     all_pos(100);
    std::vector<real_type> steps;
    for (Real3& pos : all_pos)
    {
        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(0.5);
        steps.push_back(result.distance);
        pos = geo.pos();
    }
    PRINT_EXPECTED(all_pos);
}

//---------------------------------------------------------------------------//

TEST_F(LayersTest, revolutions_through_layers)
{
    const real_type radius{3.8085385437789383};
    auto            particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10.9181415106});
    auto          geo = this->init_geo({radius, 0, 0}, {0, 1, 0});
    UniformZField field(1.0 * units::tesla);

    // Build propagator
    FieldDriverOptions driver_options;
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, &geo);

    // clang-format off
    static const real_type expected_y[]
        = { 0.5,  1.5,  2.5,  3.5,  3.5,  2.5,  1.5,  0.5,
           -0.5, -1.5, -2.5, -3.5, -3.5, -2.5, -1.5, -0.5};
    // clang-format on
    const int    num_boundary = sizeof(expected_y) / sizeof(real_type);
    const int    num_revs     = 10;
    const int    num_steps    = 100;
    const double step         = (2 * pi * radius) / num_steps;

    int       icross       = 0;
    real_type total_length = 0;

    for (CELER_MAYBE_UNUSED int ir : celeritas::range(num_revs))
    {
        for (CELER_MAYBE_UNUSED auto k : celeritas::range(num_steps))
        {
            auto result = propagate(step);
            total_length += result.distance;

            if (result.boundary)
            {
                int j = icross++ % num_boundary;
                EXPECT_DOUBLE_EQ(expected_y[j], geo.pos()[1]);
                geo.cross_boundary();
            }
        }
    }

    EXPECT_SOFT_NEAR(-0.13150565, geo.pos()[0], 1e-6);
    EXPECT_SOFT_NEAR(-0.03453068, geo.dir()[1], 1e-6);
    EXPECT_SOFT_NEAR(221.48171708, total_length, 1e-6);
    EXPECT_EQ(148, icross);
}

TEST_F(LayersTest, revolutions_through_cms_field)
{
    // Scale the test radius with the approximated center value of the
    // parameterized field (3.8 units::tesla)
    real_type radius   = 3.8085386036 / 3.8;
    auto      particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10.9181415106});
    auto geo = this->init_geo({radius, -10, 0}, {0, 1, 0});

    celeritas_test::detail::CMSParameterizedField field;
    FieldDriverOptions                            driver_options;

    EXPECT_SOFT_NEAR(
        radius, this->calc_field_curvature(particle, geo, field), 5e-3);

    // Build propagator
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, &geo);

    const int    num_revs  = 10;
    const int    num_steps = 100;
    const double step      = (2 * pi * radius) / num_steps;

    real_type total_length = 0;

    for (CELER_MAYBE_UNUSED int ir : celeritas::range(num_revs))
    {
        for (CELER_MAYBE_UNUSED auto k : celeritas::range(num_steps))
        {
            auto result = propagate(step);
            total_length += result.distance;
            EXPECT_DOUBLE_EQ(step, result.distance);
            ASSERT_FALSE(result.boundary);
            EXPECT_DOUBLE_EQ(step, result.distance);
        }
    }
    EXPECT_SOFT_NEAR(2 * pi * radius * num_revs, total_length, 1e-5);
}

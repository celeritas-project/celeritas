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
#include "corecel/math/Algorithms.hh"
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

#include "CMSParameterizedField.hh"
#include "DiagnosticStepper.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using constants::pi;
using constants::sqrt_three;
using units::MevEnergy;

template<class E>
using DiagnosticDPStepper = DiagnosticStepper<DormandPrinceStepper<E>>;

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
        using namespace units;

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
        geo_state_ = GeoStateStore(this->geometry()->host_ref(), 1);
        par_state_ = ParStateStore(this->particle()->host_ref(), 1);
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
        return {0, 0, this->strength * max<real_type>(0, pos[2])};
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
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());
    FieldDriverOptions driver_options;
    auto               propagate
        = make_field_propagator(stepper, driver_options, particle, &geo);

    // Test step that's smaller than driver's minimum (won't actually alter geo
    // state but should return the distance as if it were moved)
    Propagation result = propagate(1e-10);
    EXPECT_DOUBLE_EQ(1e-10, result.distance);
    EXPECT_FALSE(result.boundary);
    EXPECT_VEC_SOFT_EQ(Real3({radius, 0, 0}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.dir());
    EXPECT_EQ(0, stepper.count());

    // Test a short step
    result = propagate(1e-2);
    EXPECT_SOFT_EQ(1e-2, result.distance);
    EXPECT_VEC_SOFT_EQ(Real3({3.80852541539105, 0.0099999885096862, 0}),
                       geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({-0.00262567606832303, 0.999996552906651, 0}),
                       geo.dir());
    EXPECT_EQ(1, stepper.count());

    // Test the remaining quarter-turn divided into 20 steps
    {
        stepper.reset_count();
        real_type step = 0.5 * pi * radius - 1e-2;
        for (auto i : range(25))
        {
            SCOPED_TRACE(i);
            result = propagate(step / 25);
            EXPECT_SOFT_EQ(step / 25, result.distance);
            EXPECT_EQ(i + 1, stepper.count());
            EXPECT_FALSE(result.boundary)
                << "At " << geo.pos() << " along " << geo.dir();
        }
        EXPECT_SOFT_NEAR(0, distance(Real3({0, radius, 0}), geo.pos()), 1e-8);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
    }

    // Test a very long (next quarter-turn) step
    {
        SCOPED_TRACE("Quarter turn");
        stepper.reset_count();
        result = propagate(0.5 * pi * radius);
        EXPECT_SOFT_EQ(0.5 * pi * radius, result.distance);
        EXPECT_LT(distance(Real3({-radius, 0, 0}), geo.pos()), 1e-6);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, -1, 0}), geo.dir()));
        EXPECT_EQ(27, stepper.count());
    }

    // Test a ridiculously long (half-turn) step to put us back at the start
    {
        SCOPED_TRACE("Half turn");
        stepper.reset_count();
        result = propagate(pi * radius);
        EXPECT_SOFT_EQ(pi * radius, result.distance);
        EXPECT_LT(distance(Real3({radius, 0, 0}), geo.pos()), 1e-5);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, 1, 0}), geo.dir()));
        EXPECT_EQ(68, stepper.count());
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

    // Construct field (shape and magnitude shouldn't matter)
    UniformZField      field(1234.5);
    FieldDriverOptions driver_options;
    auto               stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());

    // Propagate inside box
    {
        auto geo       = this->init_geo({0, 0, 0}, {0, 0, 1});
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, &geo);

        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(3.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 3}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, stepper.count());
    }
    // Move to boundary
    {
        auto geo = this->make_geo_view();
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, &geo);

        stepper.reset_count();
        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(2.0, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 5}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(2, stepper.count());
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
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, &geo);

        stepper.reset_count();
        auto result = propagate(5.0);
        EXPECT_SOFT_EQ(5.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 10}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, stepper.count());
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

        EXPECT_FALSE(result.boundary);
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

        EXPECT_FALSE(result.boundary);
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
TEST_F(TwoBoxTest, electron_tangent_cross)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    UniformZField      field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    // Circumference
    const real_type circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits boundary");

        real_type dy = 1.1 * driver_options.delta_chord;

        auto geo       = this->init_geo({1, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x     = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({x, 5, 0}), geo.pos()), 1e-5)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 1e-5)
            << "Ending direction at " << geo.dir();

        if (!CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ("inner_box.py",
                      this->geometry()->id_to_label(geo.surface_id()));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->geometry()->id_to_label(geo.volume_id()));
        EXPECT_FALSE(geo.is_outside());
    }
    {
        SCOPED_TRACE("Barely misses boundary");

        real_type dy = 0.9 * driver_options.delta_chord;

        auto geo       = this->init_geo({1, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        EXPECT_SOFT_EQ(circ, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_LT(distance(Real3({1, 4 + dy, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({0, 1, 0}), geo.dir()), 1e-5);
    }
}

TEST_F(TwoBoxTest, electron_corner_hit)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    UniformZField      field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    // Circumference
    const real_type circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits y boundary");

        real_type dy = 1.1 * driver_options.delta_chord;

        auto geo       = this->init_geo({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x     = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5 + x, 5, 0}), geo.pos()), 1e-5)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 1e-5)
            << "Ending direction at " << geo.dir();

        if (!CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ("inner_box.py",
                      this->geometry()->id_to_label(geo.surface_id()));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->geometry()->id_to_label(geo.volume_id()));
        EXPECT_FALSE(geo.is_outside());
    }
    {
        SCOPED_TRACE("Hits y because the chord goes through x first");

        real_type dy = 0.001 * driver_options.delta_chord;

        auto geo       = this->init_geo({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x     = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5 + x, 5, 0}), geo.pos()), 1e-4)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 1e-4)
            << "Ending direction at " << geo.dir();

        if (!CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ("inner_box.py",
                      this->geometry()->id_to_label(geo.surface_id()));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->geometry()->id_to_label(geo.volume_id()));
        EXPECT_FALSE(geo.is_outside());
    }
    {
        SCOPED_TRACE("Barely (correctly) misses y");

        real_type dy = -0.001 * driver_options.delta_chord;

        auto geo       = this->init_geo({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        auto result = propagate(circ);

        EXPECT_SOFT_NEAR(circ * .25, result.distance, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5, 5 + dy, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({-1, 0, 0}), geo.dir()), 1e-5);

        if (!CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ("inner_box.mx",
                      this->geometry()->id_to_label(geo.surface_id()));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->geometry()->id_to_label(geo.volume_id()));
        EXPECT_FALSE(geo.is_outside());
    }
}

// Endpoint of a step is very close to the boundary.
TEST_F(TwoBoxTest, electron_step_endpoint)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    UniformZField      field(unit_radius_field_strength);
    FieldDriverOptions driver_options;
    driver_options.delta_intersection = 0.1;

    // First step length and position from starting at {0,0,0} along {0,1,0}
    static constexpr real_type first_step = 0.44815869703174;
    static constexpr Real3     first_pos
        = {-0.098753281951459, 0.43330671122068, 0};

    {
        SCOPED_TRACE("First step ends barely closer than boundary");
        // Note: this ends up being the !linear_step.boundary case

        real_type dx = 0.1 * driver_options.delta_intersection;
        Real3     start_pos{-5 + dx, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        auto geo       = this->init_geo(start_pos, {0, 1, 0});
        auto stepper   = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, &geo);
        auto result = propagate(first_step);

        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(3, stepper.count());
        EXPECT_SOFT_EQ(result.distance, first_step);
        EXPECT_LT(distance(Real3{-4.99000022992164, 8.24444331692931e-08, 0},
                           geo.pos()),
                  1e-8);
    }
    {
        SCOPED_TRACE("First step ends on boundary");

        real_type dx = 0;
        Real3     start_pos{-5 - dx, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        auto geo       = this->init_geo(start_pos, {0, 1, 0});
        auto stepper   = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, &geo);
        auto result = propagate(first_step);

        EXPECT_TRUE(result.boundary);
        EXPECT_EQ(3, stepper.count());
        EXPECT_SOFT_NEAR(result.distance, first_step, 1e-10);
        // Y position suffers from roundoff
        EXPECT_LT(distance(Real3{-5.0, -9.26396730438483e-07, 0}, geo.pos()),
                  1e-8);
    }
    {
        SCOPED_TRACE("First step is a slightly further than boundary");

        real_type dx = 0.1 * driver_options.delta_intersection;
        Real3     start_pos{-5 - dx, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        auto geo       = this->init_geo(start_pos, {0, 1, 0});
        auto stepper   = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, &geo);
        auto result = propagate(first_step);

        EXPECT_TRUE(result.boundary);
        EXPECT_EQ(3, stepper.count());
        EXPECT_LT(result.distance, first_step);
        EXPECT_SOFT_EQ(0.44613335936932041, result.distance);
        EXPECT_VEC_SOFT_EQ((Real3{-5, -0.0438785349441534, 0}), geo.pos());
    }
}

// Heuristic test: plotting points with finer propagation distance show a track
// with decreasing radius
TEST_F(TwoBoxTest, nonuniform_field)
{
    auto particle = this->init_particle(
        this->particle()->find(pdg::electron()), MevEnergy{10});
    ReluZField         field{unit_radius_field_strength};
    FieldDriverOptions driver_options;

    this->init_geo({-2.0, 0, 0}, {0, 1, 1});

    static const Real3 expected_all_pos[]
        = {{-2.082588410019, 0.698321021704, 0.70710499699532},
           {-2.5772835670309, 1.1563856325251, 1.414208222427},
           {-3.0638597406072, 0.77477344365218, 2.1213130872532},
           {-2.5584323246703, 0.58519068474743, 2.8284269544184},
           {-2.904435093832, 0.86378022294055, 3.5355750279272},
           {-2.5804988125119, 0.7657810943241, 4.242802666321},
           {-2.7424915491399, 0.60277842755393, 4.9501038870007},
           {-2.6941223485135, 0.6137455428308, 5}};
    for (const Real3& pos : expected_all_pos)
    {
        auto geo       = this->make_geo_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, &geo);
        propagate(1.0);
        EXPECT_VEC_SOFT_EQ(pos, geo.pos());
    }
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

    for (CELER_MAYBE_UNUSED int ir : range(num_revs))
    {
        for (CELER_MAYBE_UNUSED auto k : range(num_steps))
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

    CMSParameterizedField field;
    FieldDriverOptions    driver_options;

    EXPECT_SOFT_NEAR(
        radius, this->calc_field_curvature(particle, geo, field), 5e-3);

    // Build propagator
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, &geo);

    const int    num_revs  = 10;
    const int    num_steps = 100;
    const double step      = (2 * pi * radius) / num_steps;

    real_type total_length = 0;

    for (CELER_MAYBE_UNUSED int ir : range(num_revs))
    {
        for (CELER_MAYBE_UNUSED auto k : range(num_steps))
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

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas

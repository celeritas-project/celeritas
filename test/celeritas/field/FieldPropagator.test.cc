//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/FieldPropagator.hh"

#include <cmath>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/GenericGeoTestBase.hh"
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

class FieldPropagatorTestBase : public GenericCoreGeoTestBase
{
  public:
    using SPConstParticle = std::shared_ptr<ParticleParams const>;

    void SetUp() override;

    SPConstParticle const& particle() const
    {
        CELER_ENSURE(particle_);
        return particle_;
    }

    ParticleTrackView make_particle_view(PDGNumber pdg, MevEnergy energy)
    {
        CELER_EXPECT(pdg && energy > zero_quantity());
        ParticleId pid = this->particle()->find(pdg);
        CELER_ASSERT(pid);
        ParticleTrackView view{
            this->particle()->host_ref(), par_state_.ref(), TrackSlotId{0}};
        view = {pid, energy};
        return view;
    }

    template<class Field>
    real_type calc_field_curvature(ParticleTrackView const& particle,
                                   GeoTrackView const& geo,
                                   Field const& calc_field) const
    {
        auto field_strength = norm(calc_field(geo.pos()));
        return native_value_from(particle.momentum())
               / (std::fabs(native_value_from(particle.charge()))
                  * field_strength);
    }

    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

  private:
    //// TYPE ALIASES ////
    template<template<Ownership, MemSpace> class T>
    using HostStateStore = CollectionStateStore<T, MemSpace::host>;
    using ParStateStore = HostStateStore<ParticleStateData>;

    //// DATA ////

    std::shared_ptr<ParticleParams const> particle_;
    ParStateStore par_state_;
};

void FieldPropagatorTestBase::SetUp()
{
    // Create particle defs
    using namespace units;
    constexpr auto stable = ParticleRecord::stable_decay_constant();
    ParticleParams::Input defs
        = {{"electron",
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
    particle_ = std::make_shared<ParticleParams>(std::move(defs));

    par_state_ = ParStateStore(particle_->host_ref(), 1);
}

//---------------------------------------------------------------------------//

class TwoBoxTest : public FieldPropagatorTestBase
{
    std::string geometry_basename() const override { return "two-boxes"; }
};

class LayersTest : public FieldPropagatorTestBase
{
    std::string geometry_basename() const override { return "field-layers"; }
};

class SimpleCmsTest : public FieldPropagatorTestBase
{
    std::string geometry_basename() const override { return "simple-cms"; }
};

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//

// Field strength is zero for z <= 0, linearly increasing for z > 0 so that at
// z=1 it has a value of "strength"
struct ReluZField
{
    real_type strength;

    Real3 operator()(Real3 const& pos) const
    {
        return {0, 0, this->strength * max<real_type>(0, pos[2])};
    }
};

// sin(1/z), scaled and with multiplicative constant
struct HorribleZField
{
    real_type strength{1};
    real_type scale{1};

    Real3 operator()(Real3 const& pos) const
    {
        return {0, 0, this->strength * std::sin(this->scale / pos[2])};
    }
};

//---------------------------------------------------------------------------//
// CONSTANTS
//---------------------------------------------------------------------------//

// Field value (native units) for 10 MeV electron/positron to have a radius of
// 1 cm
constexpr real_type unit_radius_field_strength{3.5019461121752274
                                               * units::tesla};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TwoBoxTest, electron_interior)
{
    // Initialize position and direction so its curved track is centered about
    // the origin, moving counterclockwise from the right
    const real_type radius{3.8085385437789383};
    auto particle
        = this->make_particle_view(pdg::electron(), MevEnergy{10.9181415106});
    auto geo = this->make_geo_track_view({radius, 0, 0}, {0, 1, 0});
    UniformZField field(1.0 * units::tesla);

    // Check expected field curvature and geometry cell
    EXPECT_SOFT_EQ(radius, this->calc_field_curvature(particle, geo, field));
    EXPECT_EQ("inner", this->volume_name(geo));

    // Build propagator
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());
    FieldDriverOptions driver_options;
    auto propagate
        = make_field_propagator(stepper, driver_options, particle, geo);

    // Test a short step
    Propagation result = propagate(1e-2);
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

    // Test step that's smaller than driver's minimum (should take one
    // iteration in the propagator loop)
    {
        stepper.reset_count();
        result = propagate(1e-10);
        EXPECT_DOUBLE_EQ(1e-10, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_NEAR(
            Real3({3.8085385881855, -2.3814749713353e-07, 0}), geo.pos(), 1e-7);
        EXPECT_VEC_NEAR(Real3({6.2529888474538e-08, 1, 0}), geo.dir(), 1e-7);
        EXPECT_EQ(1, stepper.count());
    }
}

TEST_F(TwoBoxTest, positron_interior)
{
    // Initialize position and direction so its curved track (radius 1) is
    // centered about the origin, moving *clockwise* from the right
    const real_type radius{1.0};
    auto particle = this->make_particle_view(pdg::positron(), MevEnergy{10});
    auto geo = this->make_geo_track_view({radius, 0, 0}, {0, -1, 0});
    UniformZField field(unit_radius_field_strength);

    // Check expected field curvature
    EXPECT_SOFT_EQ(radius, this->calc_field_curvature(particle, geo, field));

    // Build propagator
    FieldDriverOptions driver_options;
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, geo);

    // Test a quarter turn
    Propagation result = propagate(0.5 * pi * radius);
    EXPECT_SOFT_EQ(0.5 * pi * radius, result.distance);
    EXPECT_SOFT_NEAR(0, distance(Real3({0, -radius, 0}), geo.pos()), 1e-5);
    EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
}

// Gamma in magnetic field should have a linear path
TEST_F(TwoBoxTest, gamma_interior)
{
    auto particle = this->make_particle_view(pdg::gamma(), MevEnergy{1});

    // Construct field (shape and magnitude shouldn't matter)
    UniformZField field(1234.5);
    FieldDriverOptions driver_options;
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());

    // Propagate inside box
    {
        auto geo = this->make_geo_track_view({0, 0, 0}, {0, 0, 1});
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);

        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(3.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 3}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, stepper.count());
    }
    // Move to boundary
    {
        auto geo = this->make_geo_track_view();
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);

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
        auto geo = this->make_geo_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    // Move in new region
    {
        auto geo = this->make_geo_track_view();
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);

        stepper.reset_count();
        auto result = propagate(5.0);
        EXPECT_SOFT_EQ(5.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 10}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, stepper.count());
    }
}

// Field really shouldn't matter to a gamma right?
TEST_F(TwoBoxTest, gamma_pathological)
{
    auto particle = this->make_particle_view(pdg::gamma(), MevEnergy{1});

    // Construct field (shape and magnitude shouldn't matter)
    HorribleZField field{1.2345 * units::tesla, 5};
    FieldDriverOptions driver_options;
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());

    // Propagate inside box
    {
        auto geo = this->make_geo_track_view({0, 0, -2}, {0, 0, 1});
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);

        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(3.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, stepper.count());
    }
}

// Gamma exits the inner volume
TEST_F(TwoBoxTest, gamma_exit)
{
    auto particle = this->make_particle_view(pdg::gamma(), MevEnergy{1});
    UniformZField field(12345.6);
    FieldDriverOptions driver_options;

    {
        SCOPED_TRACE("Exact boundary");
        auto geo = this->make_geo_track_view({2, 4.75, 0}, {0, 1, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(0.25);

        EXPECT_SOFT_EQ(0.25, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({2, 5, 0}), geo.pos()), 1e-5);
        EXPECT_EQ(1, stepper.count());
        EXPECT_EQ("inner", this->volume_name(geo));
        ASSERT_TRUE(result.boundary);
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE(
            "Reported distance is based on requested step, not actual "
            "boundary, to avoid an extra substep");
        auto geo = this->make_geo_track_view({2, 4.749, 0}, {0, 1, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(0.251 + 1e-7);

        EXPECT_SOFT_EQ(0.251, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({2, 5, 0}), geo.pos()), 1e-5);
        EXPECT_EQ(1, stepper.count());
        EXPECT_EQ("inner", this->volume_name(geo));
        ASSERT_TRUE(result.boundary);
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    for (real_type d : {0.5, 1e4})
    {
        SCOPED_TRACE("Long step");
        auto geo = this->make_geo_track_view({2, 4.749, 0}, {0, 1, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(d);

        EXPECT_SOFT_EQ(0.251, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({2, 5, 0}), geo.pos()), 1e-5);
        EXPECT_EQ(2, stepper.count());
        EXPECT_EQ("inner", this->volume_name(geo));
        ASSERT_TRUE(result.boundary);
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
}

TEST_F(TwoBoxTest, electron_super_small_step)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{2});
    UniformZField field(1 * units::tesla);
    FieldDriverOptions driver_options;
    for (real_type delta : {1e-14, 1e-8, 1e-2, 0.1})
    {
        auto geo = this->make_geo_track_view({90, 90, 90}, {1, 0, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(delta);

        EXPECT_DOUBLE_EQ(delta, result.distance);
        EXPECT_EQ(1, stepper.count());
    }
}

// Electron takes small steps up to and from a boundary
TEST_F(TwoBoxTest, electron_small_step)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;
    constexpr real_type delta = 1e-7;

    {
        SCOPED_TRACE("Small step *not quite* to boundary");

        auto geo
            = this->make_geo_track_view({5 - delta - 1.0e-5, 0, 0}, {1, 0, 0});
        EXPECT_FALSE(geo.is_on_boundary());

        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(delta);

        // Search distance doesn't hit boundary
        EXPECT_SOFT_EQ(result.distance, delta);
        EXPECT_FALSE(result.boundary);
        EXPECT_FALSE(geo.is_on_boundary());
        EXPECT_VEC_NEAR(Real3({5 - 1.0e-5, 0, 0}), geo.pos(), 1e-7);
    }
    {
        SCOPED_TRACE("Small step *almost* to boundary");

        auto geo = this->make_geo_track_view({5 - 2 * delta, 0, 0}, {1, 0, 0});
        EXPECT_FALSE(geo.is_on_boundary());

        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(delta);

        // The boundary search goes an extra driver_.delta_intersection()
        // (1e-7) past the requested end point
        EXPECT_SOFT_EQ(result.distance, delta);
        EXPECT_FALSE(result.boundary);
        EXPECT_FALSE(geo.is_on_boundary());
        EXPECT_VEC_SOFT_EQ(Real3({4.9999999, 0, 0}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({1, delta, 0}), geo.dir());
    }
    {
        SCOPED_TRACE("Small step intersected by boundary");

        auto geo = this->make_geo_track_view({5 - delta, 0, 0}, {1, 0, 0});
        EXPECT_FALSE(geo.is_on_boundary());

        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(2 * delta);

        EXPECT_LE(result.distance, 2 * delta);
        EXPECT_SOFT_NEAR(
            1.0000000044408872e-07,
            result.distance,
            (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4 ? 1e-12 : 1e-8));
        EXPECT_TRUE(result.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        EXPECT_VEC_SOFT_EQ(Real3({5, 0, 0}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({1, 2 * delta, 0}), geo.dir());
    }
    {
        SCOPED_TRACE("Cross boundary");

        auto geo = this->make_geo_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Small step from boundary");

        auto geo = this->make_geo_track_view();
        EXPECT_TRUE(geo.is_on_boundary());

        // Starting on the boundary, take a step smaller than driver's minimum
        // (could be, e.g., a very small distance to interaction)
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(delta);

        EXPECT_DOUBLE_EQ(delta, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_FALSE(geo.is_on_boundary());
        EXPECT_VEC_SOFT_EQ(Real3({5 + delta, 0, 0}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({1, 3 * delta, 0}), geo.dir());
    }
}

// Electron will be tangent to the boundary at the top of its curved path.
TEST_F(TwoBoxTest, electron_tangent)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    {
        SCOPED_TRACE("Nearly quarter turn close to boundary");

        auto geo = this->make_geo_track_view({1, 4, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
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

        auto geo = this->make_geo_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
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
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(0.5 * unit_radius_field_strength);
    FieldDriverOptions driver_options;

    {
        auto geo = this->make_geo_track_view({2, 4, 0}, {0, 1, 0});
        EXPECT_SOFT_EQ(2.0, this->calc_field_curvature(particle, geo, field));
    }
    const real_type circ = 2.0 * 2 * pi;

    {
        SCOPED_TRACE("Exit (twelfth of a turn)");

        auto geo = this->make_geo_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(pi);

        EXPECT_SOFT_NEAR(1. / 12., result.distance / circ, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({sqrt_three, 5, 0}), geo.pos()), 1e-5);
        // Direction should be up left
        EXPECT_LT(distance(Real3({-0.5, sqrt_three / 2, 0}), geo.dir()), 1e-5);
    }
    {
        SCOPED_TRACE("Cross boundary");

        auto geo = this->make_geo_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Reenter (1/3 turn)");

        auto geo = this->make_geo_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        EXPECT_SOFT_NEAR(1. / 3., result.distance / circ, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-sqrt_three, 5, 0}), geo.pos()), 1e-5);
        // Direction should be down left
        EXPECT_LT(distance(Real3({-0.5, -sqrt_three / 2, 0}), geo.dir()), 1e-5);
    }
    {
        SCOPED_TRACE("Cross boundary");

        auto geo = this->make_geo_track_view();
        geo.cross_boundary();
        EXPECT_EQ("inner", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Return to start (2/3 turn)");

        auto geo = this->make_geo_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(7. / 12. * circ);

        EXPECT_SOFT_NEAR(7. / 12., result.distance / circ, 1e-5);
        EXPECT_FALSE(result.boundary);
        EXPECT_LT(distance(Real3({2, 4, 0}), geo.pos()), 2e-5);
        EXPECT_LT(distance(Real3({0, 1, 0}), geo.dir()), 1e-5);
    }
}

// Electron barely crosses boundary
TEST_F(TwoBoxTest, electron_tangent_cross)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    // Circumference
    const real_type circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits boundary");

        real_type dy = 1.1 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({1, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({x, 5, 0}), geo.pos()), 1e-5)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 1e-5)
            << "Ending direction at " << geo.dir();

        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("inner_box.py", this->surface_name(geo));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Barely misses boundary");

        real_type dy = 0.9 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({1, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        EXPECT_SOFT_EQ(circ, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_LT(distance(Real3({1, 4 + dy, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({0, 1, 0}), geo.dir()), 1e-5);
    }
}

TEST_F(TwoBoxTest, electron_corner_hit)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;

    // Circumference
    const real_type circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits y boundary");

        real_type dy = 1.1 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5 + x, 5, 0}), geo.pos()), 1e-5)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 1e-5)
            << "Ending direction at " << geo.dir();

        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("inner_box.py", this->surface_name(geo));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Hits y because the chord goes through x first");

        real_type dy = 0.001 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5 + x, 5, 0}), geo.pos()), 1e-4)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 1e-4)
            << "Ending direction at " << geo.dir();

        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("inner_box.py", this->surface_name(geo));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Barely (correctly) misses y");

        real_type dy = -0.001 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        EXPECT_SOFT_NEAR(circ * .25, result.distance, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5, 5 + dy, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({-1, 0, 0}), geo.dir()), 1e-5);

        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("inner_box.mx", this->surface_name(geo));
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
}

// Endpoint of a step is very close to the boundary.
TEST_F(TwoBoxTest, electron_step_endpoint)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;
    driver_options.delta_intersection = 0.1;

    // First step length and position from starting at {0,0,0} along {0,1,0}
    static constexpr real_type first_step = 0.44815869703174;
    static constexpr Real3 first_pos
        = {-0.098753281951459, 0.43330671122068, 0};

    {
        SCOPED_TRACE("First step ends barely closer than boundary");
        /*
         * Note: this ends up being the !linear_step.boundary case:
          Propagate up to 0.448159
          - advance(0.348159, {-4.89125,-0.433307,0})
                -> {0.348159, {-4.95124,-0.0921392,0}}
           + chord length 0.346403 => linear step 0.446403
           + advancing to substep end point
        */

        real_type dx = 0.1 * driver_options.delta_intersection;
        Real3 start_pos{-5 + dx, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        auto geo = this->make_geo_track_view(start_pos, {0, 1, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(first_step - driver_options.delta_intersection);

        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(1, stepper.count());
        EXPECT_SOFT_EQ(first_step - driver_options.delta_intersection,
                       result.distance);
        EXPECT_LT(distance(Real3{-4.9512441890768795, -0.092139178167222446, 0},
                           geo.pos()),
                  1e-8)
            << geo.pos();
    }
    {
        SCOPED_TRACE("First step ends barely closer than boundary");
        /*
         Propagate up to 0.448159
         - advance(0.448159, {-4.89125,-0.433307,0})
           -> {0.448159, {-4.99,8.24444e-08,0}}
          + chord length 0.444418 => linear step 0.489419 (hit surface 6):
           update length 0.493539
          + next trial step exceeds driver minimum 1e-06 *OR* intercept is
           sufficiently close (miss distance = 0.0450017) to substep point
         - Moved remaining distance 0 without physically changing position
         ==> distance 0.448159 (in 0 steps)
         */

        real_type dx = 0.1 * driver_options.delta_intersection;
        Real3 start_pos{-5 + dx, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        auto geo = this->make_geo_track_view(start_pos, {0, 1, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(first_step);

        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(3, stepper.count());
        EXPECT_SOFT_EQ(0.44815869703173999, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(
            distance(Real3{-4.9900002299216384, 8.2444433238682002e-08, 0},
                     geo.pos()),
            1e-8)
            << geo.pos();
    }
    {
        SCOPED_TRACE("First step ends on boundary");

        real_type dx = 0;
        Real3 start_pos{-5 - dx, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        auto geo = this->make_geo_track_view(start_pos, {0, 1, 0});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(first_step);

        EXPECT_TRUE(result.boundary);
        EXPECT_EQ(3, stepper.count());
        EXPECT_SOFT_NEAR(result.distance, first_step, 1e-5);
        EXPECT_LT(result.distance, first_step);
        // Y position suffers from roundoff
        EXPECT_LT(distance(Real3{-5.0, -9.26396730438483e-07, 0}, geo.pos()),
                  1e-8);
    }
}

// Electron barely crosses boundary
TEST_F(TwoBoxTest, electron_tangent_cross_smallradius)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});

    UniformZField field(unit_radius_field_strength * 100);
    const real_type radius = 0.01;
    const real_type miss_distance = 1e-4;

    std::vector<int> boundary;
    std::vector<real_type> distances;
    std::vector<int> substeps;
    std::vector<std::string> volumes;

    for (real_type dtheta : {pi / 4, pi / 7, 1e-3, 1e-6, 1e-9})
    {
        SCOPED_TRACE(dtheta);
        {
            // Angle of intercept with boundary
            real_type tint = std::asin((radius - miss_distance) / radius);
            const real_type sintheta = std::sin(tint - dtheta);
            const real_type costheta = std::cos(tint - dtheta);

            Real3 pos{radius * costheta,
                      5 + miss_distance - radius + radius * sintheta,
                      0};
            Real3 dir{-sintheta, costheta, 0};
            this->make_geo_track_view(pos, dir);
        }
        auto geo = this->make_geo_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));

        EXPECT_SOFT_EQ(radius,
                       this->calc_field_curvature(particle, geo, field));

        // Build propagator
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        FieldDriverOptions driver_options;
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        for (int i : range(2))
        {
            SCOPED_TRACE(i);
            auto result = propagate(radius * dtheta);
            if (result.boundary)
            {
                geo.cross_boundary();
            }

            boundary.push_back(result.boundary);
            distances.push_back(result.distance);
            substeps.push_back(stepper.count());
            volumes.push_back(this->volume_name(geo));
            stepper.reset_count();
        }
    }

    static int const expected_boundary[] = {1, 1, 1, 1, 1, 0, 1, 0, 1, 0};
    EXPECT_VEC_EQ(expected_boundary, boundary);
    static double const expected_distances[] = {0.00785398163,
                                                0.00282334506,
                                                0.00448798951,
                                                0.00282597038,
                                                1e-05,
                                                1e-05,
                                                1e-08,
                                                1e-08,
                                                9.99379755e-12,
                                                1e-11};
    EXPECT_VEC_NEAR(expected_distances, distances, 1e-5);
    static int const expected_substeps[] = {4, 63, 3, 14, 1, 1, 1, 1, 1, 1};

    EXPECT_VEC_EQ(expected_substeps, substeps);
    static char const* expected_volumes[] = {"world",
                                             "inner",
                                             "world",
                                             "inner",
                                             "world",
                                             "world",
                                             "world",
                                             "world",
                                             "world",
                                             "world"};
    EXPECT_VEC_EQ(expected_volumes, volumes);
}

// Heuristic test: plotting points with finer propagation distance show a track
// with decreasing radius
TEST_F(TwoBoxTest, nonuniform_field)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    ReluZField field{unit_radius_field_strength};
    FieldDriverOptions driver_options;

    this->make_geo_track_view({-2.0, 0, 0}, {0, 1, 1});

    static const Real3 expected_all_pos[]
        = {{-2.082588410019, 0.698321021704, 0.70710499699532},
           {-2.5772835670309, 1.1563856325251, 1.414208222427},
           {-3.0638597406072, 0.77477344365218, 2.1213130872532},
           {-2.5584323246703, 0.58519068474743, 2.8284269544184},
           {-2.904435093832, 0.86378022294055, 3.5355750279272},
           {-2.5804988125119, 0.7657810943241, 4.242802666321},
           {-2.7424915491399, 0.60277842755393, 4.9501038870007},
           {-2.6941223485135, 0.6137455428308, 5}};
    for (Real3 const& pos : expected_all_pos)
    {
        auto geo = this->make_geo_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        propagate(1.0);
        EXPECT_VEC_SOFT_EQ(pos, geo.pos());
    }
}

//---------------------------------------------------------------------------//

TEST_F(LayersTest, revolutions_through_layers)
{
    const real_type radius{3.8085385437789383};
    auto particle
        = this->make_particle_view(pdg::electron(), MevEnergy{10.9181415106});
    auto geo = this->make_geo_track_view({radius, 0, 0}, {0, 1, 0});
    UniformZField field(1.0 * units::tesla);

    // Build propagator
    FieldDriverOptions driver_options;
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, geo);

    // clang-format off
    static const real_type expected_y[]
        = { 0.5,  1.5,  2.5,  3.5,  3.5,  2.5,  1.5,  0.5,
           -0.5, -1.5, -2.5, -3.5, -3.5, -2.5, -1.5, -0.5};
    // clang-format on
    int const num_boundary = sizeof(expected_y) / sizeof(real_type);
    int const num_revs = 10;
    int const num_steps = 100;
    double const step = (2 * pi * radius) / num_steps;

    int icross = 0;
    real_type total_length = 0;

    for ([[maybe_unused]] int ir : range(num_revs))
    {
        for ([[maybe_unused]] auto k : range(num_steps))
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
    real_type radius = 3.8085386036 / 3.8;
    auto particle
        = this->make_particle_view(pdg::electron(), MevEnergy{10.9181415106});
    auto geo = this->make_geo_track_view({radius, -10, 0}, {0, 1, 0});

    CMSParameterizedField field;
    FieldDriverOptions driver_options;

    EXPECT_SOFT_NEAR(
        radius, this->calc_field_curvature(particle, geo, field), 5e-3);

    // Build propagator
    auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
        field, driver_options, particle, geo);

    int const num_revs = 10;
    int const num_steps = 100;
    double const step = (2 * pi * radius) / num_steps;

    real_type total_length = 0;

    for ([[maybe_unused]] int ir : range(num_revs))
    {
        for ([[maybe_unused]] auto k : range(num_steps))
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

TEST_F(SimpleCmsTest, electron_stuck)
{
    auto particle = this->make_particle_view(pdg::electron(),
                                             MevEnergy{4.25402379798713e-01});
    UniformZField field(1 * units::tesla);
    FieldDriverOptions driver_options;

    auto geo = this->make_geo_track_view(
        {-2.43293925496543e+01, -1.75522265870979e+01, 2.80918346435833e+02},
        {7.01343313647855e-01, -6.43327996599957e-01, 3.06996164784077e-01});

    auto calc_radius
        = [geo]() { return std::hypot(geo.pos()[0], geo.pos()[1]); };
    EXPECT_SOFT_EQ(30.000000000000011, calc_radius());

    {
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(1000);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_EQ("si_tracker", this->volume_name(geo));
        ASSERT_TRUE(geo.is_on_boundary());
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("guide_tube.coz", this->surface_name(geo));
        }
        EXPECT_SOFT_EQ(29.999999999999996, calc_radius());
        geo.cross_boundary();
        EXPECT_EQ("vacuum_tube", this->volume_name(geo));
    }
    {
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(1000);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_LE(92, stepper.count());
        EXPECT_LE(stepper.count(), 93);
        ASSERT_TRUE(geo.is_on_boundary());
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("guide_tube.coz", this->surface_name(geo));
        }
        EXPECT_SOFT_EQ(30, calc_radius());
        geo.cross_boundary();
        EXPECT_EQ("si_tracker", this->volume_name(geo));
    }
    {
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            field, driver_options, particle, geo);
        auto result = propagate(1000);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        ASSERT_TRUE(geo.is_on_boundary());
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("guide_tube.coz", this->surface_name(geo));
            EXPECT_SOFT_EQ(30, calc_radius());
        }
        else
        {
            EXPECT_SOFT_NEAR(30, calc_radius(), 1e-5);
        }
        geo.cross_boundary();
        EXPECT_EQ("vacuum_tube", this->volume_name(geo));
    }
}

TEST_F(SimpleCmsTest, vecgeom_failure)
{
    UniformZField field(1 * units::tesla);
    FieldDriverOptions driver_options;

    auto geo = this->make_geo_track_view({1.23254142755319734e+02,
                                          -2.08186543568394598e+01,
                                          -4.08262349901495583e+01},
                                         {-2.59700373666105766e-01,
                                          -8.11661685885768147e-01,
                                          -5.23221772848529443e-01});

    auto calc_radius
        = [geo]() { return std::hypot(geo.pos()[0], geo.pos()[1]); };

    bool successful_reentry = false;
    {
        auto particle = this->make_particle_view(
            pdg::electron(), MevEnergy{3.27089632881079409e-02});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        auto result = propagate(1.39170198361108938e-05);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_EQ("em_calorimeter", this->volume_name(geo));
        EXPECT_SOFT_EQ(125.00000000000001, calc_radius());
        EXPECT_EQ(2, stepper.count());
        EXPECT_FALSE(result.looping);
    }
    {
        ASSERT_TRUE(geo.is_on_boundary());
        // Simulate MSC making us reentrant
        geo.set_dir({-1.31178657592616127e-01,
                     -8.29310561920304168e-01,
                     -5.43172303859124073e-01});
        geo.cross_boundary();
        successful_reentry = (this->volume_name(geo) == "em_calorimeter");
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // ORANGE should successfully reenter. However, under certain
            // system configurations, VecGeom will end up in the world volume,
            // so we don't test in all cases.
            EXPECT_EQ("em_calorimeter", this->volume_name(geo));
        }
    }
    {
        auto particle = this->make_particle_view(
            pdg::electron(), MevEnergy{3.25917780979408864e-02});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        // This absurdly long step is because in the "failed" case the track
        // thinks it's in the world volume (nearly vacuum)
        auto result = propagate(2.12621374950874703e+21);
        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_SOFT_NEAR(125, calc_radius(), 1e-2);
        if (successful_reentry)
        {
            // Extremely long propagation stopped by substep countdown
            EXPECT_SOFT_EQ(11.676851876556075, result.distance);
            EXPECT_EQ("em_calorimeter", this->volume_name(geo));
            EXPECT_EQ(7800, stepper.count());
            EXPECT_TRUE(result.looping);
        }
        else
        {
            // Repeated substep bisection failed; particle is bumped
            EXPECT_SOFT_EQ(1e-6, result.distance);
            // Minor floating point differences could make this 102 or 103
            EXPECT_SOFT_NEAR(real_type(103), real_type(stepper.count()), 0.02);
            EXPECT_FALSE(result.looping);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

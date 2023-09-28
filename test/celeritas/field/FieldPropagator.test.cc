//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/FieldPropagator.hh"

#include <cmath>
#include <regex>

#include "celeritas_cmake_strings.h"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/geo/CheckedGeoTrackView.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "../GenericGeoTestBase.hh"
#include "CMSParameterizedField.hh"
#include "DiagnosticGeoTrackView.hh"
#include "DiagnosticStepper.hh"
#include "FieldTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using constants::pi;
using constants::sqrt_three;

template<class E>
using DiagnosticDPStepper = DiagnosticStepper<DormandPrinceStepper<E>>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldPropagatorTestBase : public GenericCoreGeoTestBase,
                                public FieldTestBase
{
    using GCGBase = GenericCoreGeoTestBase;
    using FBase = FieldTestBase;

  public:
    //!@{
    //! \name Type aliases
    using CGeoTrackView
        = CheckedGeoTrackView<GenericCoreGeoTestBase::GeoTrackView>;
    //!@}

  protected:
    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

    //! Get a single-thread host track view
    CGeoTrackView make_geo_track_view()
    {
        return CGeoTrackView{GCGBase::make_geo_track_view()};
    }

    //! Get and initialize a single-thread host track view
    CGeoTrackView make_geo_track_view(Real3 const& pos, Real3 dir)
    {
        return CGeoTrackView{GCGBase::make_geo_track_view(pos, dir)};
    }

    SPConstParticle build_particle() const final;
};

//---------------------------------------------------------------------------//
auto FieldPropagatorTestBase::build_particle() const -> SPConstParticle
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
    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//

class TwoBoxesTest : public FieldPropagatorTestBase
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

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    define CmseTest DISABLED_CmseTest
#endif
class CmseTest : public FieldPropagatorTestBase
{
    std::string geometry_basename() const override { return "cmse"; }
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

TEST_F(TwoBoxesTest, electron_interior)
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
    EXPECT_EQ(0, geo.intersect_count());
    EXPECT_EQ(1, geo.safety_count());

    // Test the remaining quarter-turn divided into 20 steps
    {
        geo.reset_count();
        stepper.reset_count();
        real_type step = 0.5 * pi * radius - 1e-2;
        for (auto i : range(25))
        {
            SCOPED_TRACE(i);
            result = propagate(step / 25);
            EXPECT_SOFT_EQ(step / 25, result.distance);
            EXPECT_EQ(i + 1, stepper.count());
            EXPECT_EQ(0, geo.intersect_count());
            EXPECT_FALSE(result.boundary)
                << "At " << geo.pos() << " along " << geo.dir();
        }
        EXPECT_SOFT_NEAR(0, distance(Real3({0, radius, 0}), geo.pos()), 1e-8);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
    }

    // Test a very long (next quarter-turn) step
    {
        SCOPED_TRACE("Quarter turn");
        geo.reset_count();
        stepper.reset_count();
        result = propagate(0.5 * pi * radius);
        EXPECT_SOFT_EQ(0.5 * pi * radius, result.distance);
        EXPECT_LT(distance(Real3({-radius, 0, 0}), geo.pos()), 1e-6);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, -1, 0}), geo.dir()));
        EXPECT_EQ(21, stepper.count());
        EXPECT_EQ(0, geo.intersect_count());
        EXPECT_EQ(6, geo.safety_count());
    }

    // Test a ridiculously long (half-turn) step to put us back at the start
    {
        SCOPED_TRACE("Half turn");
        geo.reset_count();
        stepper.reset_count();
        result = propagate(pi * radius);
        EXPECT_SOFT_EQ(pi * radius, result.distance);
        EXPECT_LT(distance(Real3({radius, 0, 0}), geo.pos()), 1e-5);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, 1, 0}), geo.dir()));
        EXPECT_EQ(40, stepper.count());
        EXPECT_EQ(0, geo.intersect_count());
        EXPECT_EQ(12, geo.safety_count());
    }

    // Test step that's smaller than driver's minimum (should take one
    // iteration in the propagator loop)
    {
        stepper.reset_count();
        geo.reset_count();
        result = propagate(1e-10);
        EXPECT_DOUBLE_EQ(1e-10, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_NEAR(
            Real3({3.8085385881855, -2.381487075086e-07, 0}), geo.pos(), 1e-7);
        EXPECT_VEC_NEAR(Real3({6.25302065531623e-08, 1, 0}), geo.dir(), 1e-7);
        EXPECT_EQ(1, stepper.count());
        EXPECT_EQ(0, geo.intersect_count());
        EXPECT_EQ(1, geo.safety_count());
    }
}

TEST_F(TwoBoxesTest, positron_interior)
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
    EXPECT_NEAR(0, distance(Real3({0, -radius, 0}), geo.pos()), 1e-5);
    EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
}

// Gamma in magnetic field should have a linear path
TEST_F(TwoBoxesTest, gamma_interior)
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
TEST_F(TwoBoxesTest, gamma_pathological)
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
TEST_F(TwoBoxesTest, gamma_exit)
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

TEST_F(TwoBoxesTest, electron_super_small_step)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{2});
    UniformZField field(1 * units::tesla);
    FieldDriverOptions driver_options;
    for (real_type delta : {1e-14, 1e-8, 1e-2, 0.1})
    {
        auto geo = this->make_geo_track_view({9.5, 9.5, 9.5}, {1, 0, 0});
        EXPECT_EQ("world", this->volume_name(geo));
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
TEST_F(TwoBoxesTest, electron_small_step)
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
        EXPECT_LT(distance(Real3({5 + delta, 0, 0}), geo.pos()), 1e-12);
        EXPECT_LT(distance(Real3({1, 3 * delta, 0}), geo.dir()), 1e-12);
    }
}

// Electron will be tangent to the boundary at the top of its curved path.
TEST_F(TwoBoxesTest, electron_tangent)
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
            2e-6);
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
            2e-6);
    }
}

// Electron crosses and reenters
TEST_F(TwoBoxesTest, electron_cross)
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
TEST_F(TwoBoxesTest, electron_tangent_cross)
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
        EXPECT_LT(distance(Real3({1, 4 + dy, 0}), geo.pos()), 2e-5);
        EXPECT_LT(distance(Real3({0, 1, 0}), geo.dir()), 1e-5);
    }
}

TEST_F(TwoBoxesTest, electron_corner_hit)
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
        EXPECT_EQ(10, geo.intersect_count());
        EXPECT_EQ(11, geo.safety_count());
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
        EXPECT_EQ(73, geo.intersect_count());
        EXPECT_EQ(74, geo.safety_count());
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
        EXPECT_EQ(5, geo.intersect_count());
        EXPECT_EQ(6, geo.safety_count());
    }
}

// Endpoint of a step is very close to the boundary.
TEST_F(TwoBoxesTest, electron_step_endpoint)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;
    const real_type dr = 0.1;
    driver_options.delta_intersection = 0.1;

    // First step length and position from starting at {0,0,0} along {0,1,0}
    static constexpr real_type first_step = 0.44815869703174;
    static constexpr Real3 first_pos
        = {-0.098753281951459, 0.43330671122068, 0};

    auto geo = this->make_geo_track_view();
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());
    auto propagate = [&](real_type start_delta, real_type move_delta) {
        Real3 start_pos{-5 + start_delta, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        stepper.reset_count();
        geo.reset_count();
        geo = GeoTrackInitializer{start_pos, {0, 1, 0}};
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);
        return propagate(first_step - move_delta);
    };

    {
        SCOPED_TRACE("First step misses boundary");
        /*
         * Note: this ends up being the !linear_step.boundary case:
         Propagate up to 0.348159 from {-4.89125,-0.433307,0} along {0,1,0}
         - advance(0.348159, {-4.89125,-0.433307,0})
           -> {0.348159, {-4.95124,-0.0921392,0}}
          + chord 0.346403 cm along {-0.173201,0.984886,0}
            => linear step 0.446403: update length 0.448666
          + advancing to substep end point (99 remaining)
         ==> distance 0.348159 (in 1 steps)
         */

        auto result = propagate(0.1 * dr, dr);
        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(1, stepper.count());
        EXPECT_SOFT_EQ(first_step - dr, result.distance);
        EXPECT_LT(distance(Real3{-4.9512441890768795, -0.092139178167222446, 0},
                           geo.pos()),
                  1e-8)
            << geo.pos();
        EXPECT_EQ(1, geo.intersect_count());
        EXPECT_EQ(1, geo.safety_count());
    }
    {
        SCOPED_TRACE("First step ends barely before boundary");
        /*
         Propagate up to 0.448159 from {-4.89125,-0.433307,0} along {0,1,0}
         - advance(0.448159, {-4.89125,-0.433307,0})
           -> {0.448159, {-4.99,3.0686e-15,0}}
          + chord 0.444418 cm along {-0.222208,0.974999,0}
            => linear step 0.48942 (HIT!): update length 0.49354
          + intercept {-5,0.0438777,0} is within 0.1 of substep endpoint
          + but it's is past the end of the step by 0.0453817
          + moved to {-4.99,3.0686e-15,0}: ignoring intercept!
         ==> distance 0.448159 (in 0 steps)
         */
        auto result = propagate(0.1 * dr, 0);
        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(1, stepper.count());
        EXPECT_SOFT_EQ(0.44815869703173999, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(-5.0, geo.pos()[0]);
        EXPECT_LT(
            distance(Real3{-4.9900002299216384, 8.2444433238682002e-08, 0},
                     geo.pos()),
            1e-6)
            << geo.pos();
        EXPECT_EQ(1, geo.intersect_count());
        EXPECT_EQ(1, geo.safety_count());
    }
    {
        SCOPED_TRACE("First step ends BARELY before boundary");
        /*
         Propagate up to 0.448159 from {-4.90125,-0.433307,0} along {0,1,0}
         - advance(0.448159, {-4.90125,-0.433307,0})
           -> {0.448159, {-5,3.0686e-15,0}}
          + chord 0.444418 cm along {-0.222208,0.974999,0}
            => linear step 0.444418 (HIT!): update length 0.448159
          + intercept {-5,4.38777e-07,0} is within 0.1 of substep endpoint
          + but it's is past the end of the step by 4.53817e-07
          + moved to {-5,3.0686e-15,0}: ignoring intercept!
         ==> distance 0.448159 (in 0 steps)
         */
        auto result = propagate(1e-6 * dr, 0);
        EXPECT_FALSE(result.boundary);
        EXPECT_EQ(1, stepper.count());
        EXPECT_SOFT_EQ(0.44815869703173999, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(-5.0, geo.pos()[0]);
        EXPECT_LT(
            distance(Real3{-4.9999998999999997, 3.0685999199146494e-15, 0},
                     geo.pos()),
            1e-6)
            << geo.pos();
        EXPECT_EQ(1, geo.intersect_count());
        EXPECT_EQ(1, geo.safety_count());
    }
    {
        SCOPED_TRACE("First step ends barely past boundary");
        /*
         Propagate up to 0.448159 from {-4.91125,-0.433307,0} along {0,1,0}
         - advance(0.448159, {-4.91125,-0.433307,0})
           -> {0.448159, {-5.01,3.0686e-15,0}}
          + chord 0.444418 cm along {-0.222208,0.974999,0}
            => linear step 0.399415 (HIT!): update length 0.402777
          + intercept {-5,-0.0438777,0} is within 0.1 of substep endpoint
         - Moved to boundary 6 at position {-5,-0.0438777,0}
         ==> distance 0.402777 (in 0 steps)
        */

        auto result = propagate(-0.1 * dr, 0);
        EXPECT_TRUE(result.boundary);
        EXPECT_EQ(1, stepper.count());
        EXPECT_SOFT_EQ(0.40277704609562048, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(distance(Real3{-5, -0.04387770235662955, 0}, geo.pos()), 1e-6)
            << geo.pos();
        EXPECT_EQ(1, geo.intersect_count());
        EXPECT_EQ(1, geo.safety_count());
    }
    {
        SCOPED_TRACE("First step ends BARELY past boundary");
        /*
         Propagate up to 0.448159 from {-4.90125,-0.433307,0} along {0,1,0}
         - advance(0.448159, {-4.90125,-0.433307,0})
           -> {0.448159, {-5,3.0686e-15,0}}
          + chord 0.444418 cm along {-0.222208,0.974999,0}
            => linear step 0.444417 (HIT!): update length 0.448158
          + intercept {-5,-4.38777e-07,0} is within 0.1 of substep endpoint
         - Moved to boundary 6 at position {-5,-4.38777e-07,0}
         */
        auto result = propagate(-1e-6 * dr, 0);
        EXPECT_TRUE(result.boundary);
        EXPECT_EQ(1, stepper.count());
        EXPECT_SOFT_EQ(0.44815824321522935, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(distance(Real3{-5, -4.3877702173875065e-07, 0}, geo.pos()),
                  1e-6)
            << geo.pos();
        EXPECT_EQ(1, geo.intersect_count());
        EXPECT_EQ(1, geo.safety_count());
    }
}

// Electron barely crosses boundary
TEST_F(TwoBoxesTest, electron_tangent_cross_smallradius)
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
    static double const expected_distances[] = {0.0078534718906499,
                                                0.0028235332722979,
                                                0.0044879852658442,
                                                0.0028259738005751,
                                                1e-05,
                                                1e-05,
                                                9.9999658622419e-09,
                                                1e-08,
                                                9.9981633254417e-12,
                                                1e-11};
    EXPECT_VEC_NEAR(expected_distances, distances, 1e-5);

    static int const expected_substeps[] = {1, 25, 1, 12, 1, 1, 1, 1, 1, 1};

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
TEST_F(TwoBoxesTest, nonuniform_field)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    ReluZField field{unit_radius_field_strength};
    FieldDriverOptions driver_options;

    auto geo = this->make_geo_track_view({-2.0, 0, 0}, {0, 1, 1});
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());
    auto propagate
        = make_field_propagator(stepper, driver_options, particle, geo);

    std::vector<real_type> all_pos;
    std::vector<int> step_counter;
    for ([[maybe_unused]] auto i : range(8))
    {
        stepper.reset_count();
        propagate(1.0);
        all_pos.insert(all_pos.end(), geo.pos().begin(), geo.pos().end());
        step_counter.push_back(stepper.count());
    }

    // clang-format off
    static double const expected_all_pos[] = {-2.0825709359803,
        0.69832583461676, 0.70710666844698, -2.5772824508968, 1.1564020888258,
        1.4141930958099, -3.0638510057122, 0.77473521479087, 2.1212684403177,
        -2.5583489716647, 0.58538451986626, 2.828330789556, -2.904690468079,
        0.86312828878343, 3.5354504022784, -2.5810333947926, 0.76746526072066,
        4.2427268982429, -2.7387860743405, 0.6033460543227, 4.9501275639478,
        -2.6908723120116, 0.6155217193027, 5};
    static int const expected_step_counter[] = {3, 3, 6, 6, 9, 10, 13, 8};
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_all_pos, all_pos);
    EXPECT_VEC_EQ(expected_step_counter, step_counter);
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
        auto result = propagate(30);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_SOFT_NEAR(double{30}, static_cast<double>(stepper.count()), 0.2);
        ASSERT_TRUE(geo.is_on_boundary());
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ("guide_tube.coz", this->surface_name(geo));
        }
        EXPECT_SOFT_EQ(30, calc_radius());
        geo.cross_boundary();
        EXPECT_EQ("si_tracker", this->volume_name(geo));
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
        ScopedLogStorer scoped_log_{&celeritas::self_logger()};
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

            // This message comes from the CheckedGeoTrackView
            static char const* const expected_log_messages[]
                = {"Volume did not change from 3 when crossing boundary at "
                   "{123.254,-20.8187,-40.8262}"};
            EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
            static char const* const expected_log_levels[] = {"warning"};
            EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
        }
        else if (!successful_reentry)
        {
            // This happens in Geant4 and *sometimes* in vecgeom
            CELER_LOG(warning) << "Reentry failed for " << celeritas_core_geo
                               << " geometry: post-propagation volume is "
                               << this->volume_name(geo);
        }
    }
    {
        ScopedLogStorer scoped_log_{&celeritas::self_logger()};
        auto particle = this->make_particle_view(
            pdg::electron(), MevEnergy{3.25917780979408864e-02});
        auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);

        Propagation result;
        // This absurdly long step is because in the "failed" case the
        // track thinks it's in the world volume (nearly vacuum)
        result = propagate(2.12621374950874703e+21);

        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
            && result.boundary != geo.is_on_boundary())
        {
            // FIXME: see #882
            GTEST_SKIP() << "The current fix fails with the Geant4 navigator";
        }

        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_SOFT_NEAR(125, calc_radius(), 1e-2);
        if (successful_reentry)
        {
            // ORANGE and *sometimes* vecgeom/geant4: extremely long
            // propagation stopped by substep countdown
            EXPECT_FALSE(result.boundary);
            EXPECT_TRUE(result.looping);
            EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;

            EXPECT_SOFT_EQ(12.02714054426572, result.distance);
            EXPECT_EQ("em_calorimeter", this->volume_name(geo));
            EXPECT_EQ(573, stepper.count());
            EXPECT_TRUE(result.looping);
        }
        else
        {
            // Repeated substep bisection failed; particle is bumped
            EXPECT_SOFT_NEAR(1e-6, result.distance, 1e-8);
            // Minor floating point differences could make this 98 or so
            EXPECT_SOFT_NEAR(real_type(95), real_type(stepper.count()), 0.05);
            EXPECT_FALSE(result.boundary);  // FIXME: should have reentered
            EXPECT_FALSE(result.looping);

            if (scoped_log_.empty()) {}
            else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
            {
                static char const* const expected_log_levels[] = {"error"};
                EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels())
                    << scoped_log_;
            }
            else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
            {
                static char const* const expected_log_messages[]
                    = {"Moved internally from boundary but safety didn't "
                       "increase: volume 6 from {123.254,-20.8187,-40.8262} "
                       "to {123.254,-20.8187,-40.8262} (distance: 1e-06)"};
                EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
                static char const* const expected_log_levels[] = {"warning"};
                EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
            }
            else
            {
                ADD_FAILURE() << "Logged warning/error:" << scoped_log_;
            }
        }
    }
}

TEST_F(CmseTest, coarse)
{
    // Build propagator
    UniformZField field{0};
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    auto stepper = make_mag_field_stepper<DiagnosticDPStepper>(
        field, particle.charge());

    FieldDriverOptions driver_options;
    driver_options.delta_intersection = 0.001;
    driver_options.delta_chord = 0.1;

    std::vector<int> num_boundary;
    std::vector<int> num_step;
    std::vector<int> num_intercept;
    std::vector<int> num_integration;

    ScopedLogStorer scoped_log_{&celeritas::self_logger()};
    bool failed{false};

    for (real_type radius : {5, 10, 20, 50})
    {
        auto geo = this->make_geo_track_view({2 * radius + 0.01, 0, -300},
                                             {0, 1, 1});
        field = UniformZField(unit_radius_field_strength / radius);
        EXPECT_SOFT_EQ(radius,
                       this->calc_field_curvature(particle, geo, field));

        auto propagate
            = make_field_propagator(stepper, driver_options, particle, geo);

        int step_count = 0;
        int boundary_count = 0;
        int const max_steps = 10000;
        while (!geo.is_outside() && step_count++ < max_steps)
        {
            Propagation result;
            try
            {
                result = propagate(radius);
            }
            catch (RuntimeError const& e)
            {
                // Failure during Geant4 propagation
                CELER_LOG(error) << e.what();
                failed = true;
                break;
            }
            if (result.boundary)
            {
                geo.cross_boundary();
                ++boundary_count;
            }
        }
        num_boundary.push_back(boundary_count);
        num_step.push_back(step_count);
        num_intercept.push_back(geo.intersect_count());
        num_integration.push_back(stepper.count());
        stepper.reset_count();
    }

    std::vector<int> expected_num_boundary = {134, 100, 60, 40};
    std::vector<int> expected_num_step = {10001, 6450, 3236, 1303};
    std::vector<int> expected_num_intercept = {30419, 19521, 16170, 9956};
    std::vector<int> expected_num_integration = {80659, 58204, 41914, 26114};

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4 && failed)
    {
        // FIXME: this happens because of incorrect momentum update
        expected_num_boundary = {134, 37, 60, 40};
        expected_num_step = {10001, 179, 3236, 1303};
        expected_num_intercept = {30419, 615, 16170, 9956};
        expected_num_integration = {80659, 1670, 41914, 26114};
    }
    else if (!scoped_log_.empty())
    {
        // Bumped (platform-dependent!): counts change a bit
        expected_num_boundary = {134, 101, 60, 40};
        expected_num_step = {10001, 6462, 3236, 1303};
        expected_num_intercept = {30419, 19551, 16170, 9956};
        expected_num_integration = {80659, 58282, 41914, 26114};
        static char const* const expected_log_messages[]
            = {"Moved internally from boundary but safety didn't increase: "
               "volume 18 from {10.3161,-6.56495,796.923} to "
               "{10.3162,-6.56497,796.923} (distance: 0.0001)"};
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages())
            << scoped_log_;
    }
    EXPECT_VEC_EQ(expected_num_boundary, num_boundary);
    EXPECT_VEC_EQ(expected_num_step, num_step);
    EXPECT_VEC_EQ(expected_num_intercept, num_intercept);
    EXPECT_VEC_EQ(expected_num_integration, num_integration);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

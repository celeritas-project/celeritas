//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/field/FieldPropagator.hh"

#include <cmath>
#include <regex>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/CheckedGeoTrackView.hh"
#include "celeritas/Constants.hh"
#include "celeritas/CoreGeoTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "CMSParameterizedField.hh"
#include "DiagnosticIntegrator.hh"
#include "FieldTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using namespace celeritas::units::literals;
//---------------------------------------------------------------------------//
constexpr real_type pi{constants::pi};
constexpr real_type sqrt_three{constants::sqrt_three};

template<class E>
using DiagnosticDPIntegrator = DiagnosticIntegrator<DormandPrinceIntegrator<E>>;

constexpr bool using_surface_vg = CELERITAS_VECGEOM_SURFACE
                                  && CELERITAS_CORE_GEO
                                         == CELERITAS_CORE_GEO_VECGEOM;

constexpr bool using_solids_vg = !CELERITAS_VECGEOM_SURFACE
                                 && CELERITAS_CORE_GEO
                                        == CELERITAS_CORE_GEO_VECGEOM;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class FieldPropagatorTestBase : public CoreGeoTestBase, public FieldTestBase
{
  protected:
    //! Get and initialize a single-thread host track view
    CheckedGeoTrackView make_geo_track_view(Real3 const& pos, Real3 const& dir)
    {
        auto result = this->make_checked_track_view();
        result = this->make_initializer(pos, dir);
        return result;
    }

    SPConstParticle build_particle() const final;
};

//---------------------------------------------------------------------------//
auto FieldPropagatorTestBase::build_particle() const -> SPConstParticle
{
    // Create particle defs
    using namespace constants;
    using namespace units;
    ParticleParams::Input defs = {{"electron",
                                   pdg::electron(),
                                   MevMass{0.5109989461},
                                   ElementaryCharge{-1},
                                   stable_decay_constant},
                                  {"positron",
                                   pdg::positron(),
                                   MevMass{0.5109989461},
                                   ElementaryCharge{1},
                                   stable_decay_constant},
                                  {"gamma",
                                   pdg::gamma(),
                                   zero_quantity(),
                                   zero_quantity(),
                                   stable_decay_constant}};
    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//

class TwoBoxesTest : public FieldPropagatorTestBase
{
    std::string_view gdml_basename() const override { return "two-boxes"; }
};

class LayersTest : public FieldPropagatorTestBase
{
    std::string_view gdml_basename() const override { return "field-layers"; }
};

class SimpleCmsTest : public FieldPropagatorTestBase
{
    std::string_view gdml_basename() const override { return "simple-cms"; }
};

#define CmseTest TEST_IF_CELERITAS_GEANT(CmseTest)
class CmseTest : public FieldPropagatorTestBase
{
    std::string_view gdml_basename() const override { return "cmse"; }
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
constexpr real_type unit_radius_field_strength{3.5019461121752274_T};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TwoBoxesTest, electron_interior)
{
    // Initialize position and direction so its curved track is centered about
    // the origin, moving counterclockwise from the right
    real_type const radius{3.8085385437789383};
    auto particle
        = this->make_particle_view(pdg::electron(), MevEnergy{10.9181415106});
    auto geo = this->make_geo_track_view({radius, 0, 0}, {0, 1, 0});
    UniformZField field(1.0_T);

    // Check expected field curvature and geometry cell
    EXPECT_SOFT_EQ(radius, this->calc_field_curvature(particle, geo, field));
    EXPECT_EQ("inner", this->volume_name(geo));

    // Build propagator
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        field, particle.charge());
    FieldDriverOptions driver_options;
    auto propagate
        = make_field_propagator(integrate, driver_options, particle, geo);

    // Test a short step
    Propagation result = propagate(1e-2);
    EXPECT_SOFT_EQ(1e-2, result.distance);
    EXPECT_VEC_SOFT_EQ(Real3({3.80852541539105, 0.0099999885096862, 0}),
                       geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({-0.00262567606832303, 0.999996552906651, 0}),
                       geo.dir());
    EXPECT_EQ(1, integrate.exchange_count());
    EXPECT_EQ(1, geo.intersect_count());
    EXPECT_EQ(0, geo.safety_count());

    // Test the remaining quarter-turn divided into 25 steps
    {
        geo.reset_count();
        real_type step = 0.5 * pi * radius - 1e-2;
        for (auto i : range(25))
        {
            SCOPED_TRACE(i);
            result = propagate(step / 25);
            EXPECT_SOFT_EQ(step / 25, result.distance);
            EXPECT_EQ(1, integrate.exchange_count());
            EXPECT_FALSE(result.boundary)
                << "At " << geo.pos() << " along " << geo.dir();
        }
        EXPECT_SOFT_NEAR(
            0, distance(Real3({0, radius, 0}), geo.pos()), coarse_eps);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({-1, 0, 0}), geo.dir()));
    }

    // Test a very long (next quarter-turn) step
    {
        SCOPED_TRACE("Quarter turn");
        result = propagate(0.5 * pi * radius);
        EXPECT_SOFT_EQ(0.5 * pi * radius, result.distance);
        EXPECT_LT(distance(Real3({-radius, 0, 0}), geo.pos()), coarse_eps);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, -1, 0}), geo.dir()));
        EXPECT_EQ(21, integrate.exchange_count());
    }

    // Test a ridiculously long (half-turn) step to put us back at the start
    {
        SCOPED_TRACE("Half turn");
        result = propagate(pi * radius);
        // The maximum substep limit in the field propagator was reached before
        // traveling the full distance; propagate again to reach the end
        real_type partial_distance = 8.7323805094658429;
        EXPECT_SOFT_EQ(partial_distance, result.distance);
        result = propagate(pi * radius - partial_distance);
        EXPECT_SOFT_EQ(pi * radius - partial_distance, result.distance);
        EXPECT_LT(distance(Real3({radius, 0, 0}), geo.pos()), 1e-5);
        EXPECT_SOFT_EQ(1.0, dot_product(Real3({0, 1, 0}), geo.dir()));
        EXPECT_EQ(40, integrate.exchange_count());
    }

    // Test step that's smaller than driver's minimum (should take one
    // iteration in the propagator loop)
    {
        result = propagate(1e-10);
        EXPECT_REAL_EQ(1e-10, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_NEAR(Real3({3.8085385881855, -2.381487075086e-07, 0}),
                        geo.pos(),
                        coarse_eps);
        EXPECT_VEC_NEAR(
            Real3({6.25302065531623e-08, 1, 0}), geo.dir(), coarse_eps);
        EXPECT_EQ(1, integrate.exchange_count());
    }
}

TEST_F(TwoBoxesTest, positron_interior)
{
    // Initialize position and direction so its curved track (radius 1) is
    // centered about the origin, moving *clockwise* from the right
    real_type const radius{1.0};
    auto particle = this->make_particle_view(pdg::positron(), MevEnergy{10});
    auto geo = this->make_geo_track_view({radius, 0, 0}, {0, -1, 0});
    UniformZField field(unit_radius_field_strength);

    // Check expected field curvature
    EXPECT_SOFT_EQ(radius, this->calc_field_curvature(particle, geo, field));

    // Build propagator
    FieldDriverOptions driver_options;
    auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        field, particle.charge());

    // Propagate inside box
    {
        auto geo = this->make_geo_track_view({0, 0, 0}, {0, 0, 1});
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);

        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(3.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 3}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, integrate.exchange_count());
    }
    // Move to boundary
    {
        auto geo = this->make_checked_track_view();
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);

        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(2.0, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 5}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(2, integrate.exchange_count());
    }
    // Cross boundary
    {
        auto geo = this->make_checked_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    // Move in new region
    {
        auto geo = this->make_checked_track_view();
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);

        auto result = propagate(5.0);
        EXPECT_SOFT_EQ(5.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 10}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, integrate.exchange_count());
    }
}

// Field really shouldn't matter to a gamma right?
TEST_F(TwoBoxesTest, gamma_pathological)
{
    auto particle = this->make_particle_view(pdg::gamma(), MevEnergy{1});

    // Construct field (shape and magnitude shouldn't matter)
    HorribleZField field{1.2345_T, 5};
    FieldDriverOptions driver_options;
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        field, particle.charge());

    // Propagate inside box
    {
        auto geo = this->make_geo_track_view({0, 0, -2}, {0, 0, 1});
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);

        auto result = propagate(3.0);
        EXPECT_SOFT_EQ(3.0, result.distance);
        EXPECT_FALSE(result.boundary);
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.pos());
        EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.dir());
        EXPECT_EQ(1, integrate.exchange_count());
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
        real_type const exact_distance = [&geo] {
            // Note: exact distance may be slightly off for VecGeom surface,
            // which applies rotation matrices to planar surfaces
            auto result = geo.find_next_step();
            EXPECT_TRUE(result.boundary);
            EXPECT_SOFT_EQ(result.distance, 0.25);
            return result.distance;
        }();

        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        auto result = propagate(exact_distance);

        EXPECT_SOFT_EQ(exact_distance, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({2, 5, 0}), geo.pos()), 1e-5);
        EXPECT_EQ(1, integrate.exchange_count());
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
        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        auto result = propagate(0.251 + 1e-7);

        EXPECT_SOFT_EQ(0.251, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({2, 5, 0}), geo.pos()), 1e-5);
        EXPECT_EQ(1, integrate.exchange_count());
        EXPECT_EQ("inner", this->volume_name(geo));
        ASSERT_TRUE(result.boundary);
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    for (real_type d : {0.5, 1e4})
    {
        SCOPED_TRACE("Long step");
        auto geo = this->make_geo_track_view({2, 4.749, 0}, {0, 1, 0});
        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        auto result = propagate(d);

        EXPECT_SOFT_EQ(0.251, result.distance);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({2, 5, 0}), geo.pos()), 1e-5);
        EXPECT_EQ(2, integrate.exchange_count());
        EXPECT_EQ("inner", this->volume_name(geo));
        ASSERT_TRUE(result.boundary);
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
}

TEST_F(TwoBoxesTest, electron_super_small_step)
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{2});
    UniformZField field(static_cast<real_type>(1_T));
    FieldDriverOptions driver_options;

    std::vector<real_type> intersect_distance;
    for (real_type delta : {1e-20, 1e-14, 1e-8, 1e-2, 0.1})
    {
        {
            SCOPED_TRACE("Far from boundary");
            auto geo = this->make_geo_track_view({9.5, 9.5, 9.5}, {1, 0, 0});
            EXPECT_EQ("world", this->volume_name(geo));
            auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
                field, particle.charge());
            auto propagate = make_field_propagator(
                integrate, driver_options, particle, geo);
            auto result = propagate(delta);

            EXPECT_REAL_EQ(delta, result.distance);
            EXPECT_EQ(1, integrate.exchange_count());
        }

        {
            SCOPED_TRACE("Bump distance intersects boundary");
            real_type const bump_distance
                = (driver_options.delta_intersection * real_type{0.1});
            real_type const eps = bump_distance * real_type{0.99};
            auto geo = this->make_geo_track_view({real_type{5.0} + eps, 0, 0},
                                                 {-1, 0, 0});
            EXPECT_EQ("world", this->volume_name(geo));
            auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
                field, particle.charge());
            auto propagate = make_field_propagator(
                integrate, driver_options, particle, geo);
            auto result = propagate(delta);

            intersect_distance.push_back(result.distance);
            EXPECT_EQ(1, integrate.exchange_count());
        }
    }
    static real_type const expected_intersect_distance[]
        = {1e-20, 1e-14, 1e-08, 9.9002453648129e-07, 9.924578491937e-07};
    EXPECT_VEC_SOFT_EQ(expected_intersect_distance, intersect_distance);
}

// Electron takes small steps up to and from a boundary
TEST_F(TwoBoxesTest, TEST_IF_CELERITAS_DOUBLE(electron_small_step))
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

        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(delta);

        // Search distance doesn't hit boundary
        EXPECT_SOFT_EQ(result.distance, delta);
        EXPECT_FALSE(result.boundary);
        EXPECT_FALSE(geo.is_on_boundary());
        EXPECT_VEC_NEAR(Real3({5 - 1.0e-5, 0, 0}), geo.pos(), coarse_eps);
    }
    {
        SCOPED_TRACE("Small step *almost* to boundary");

        auto geo = this->make_geo_track_view({5 - 2 * delta, 0, 0}, {1, 0, 0});
        EXPECT_FALSE(geo.is_on_boundary());

        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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

        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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

        auto geo = this->make_checked_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Small step from boundary");

        auto geo = this->make_checked_track_view();
        EXPECT_TRUE(geo.is_on_boundary());

        // Starting on the boundary, take a step smaller than driver's minimum
        // (could be, e.g., a very small distance to interaction)
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(delta);

        EXPECT_REAL_EQ(delta, result.distance);
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

        constexpr real_type quarter = 0.49 * pi;
        auto geo = this->make_geo_track_view({1, 4, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(quarter);

        EXPECT_FALSE(result.boundary);
        EXPECT_SOFT_EQ(quarter, result.distance);
        EXPECT_LT(distance(Real3({std::cos(quarter), 4 + std::sin(quarter), 0}),
                           geo.pos()),
                  real_type{2e-6});
    }
    {
        SCOPED_TRACE("Short step tangent to boundary");

        constexpr real_type quarter = 0.51 * pi;
        auto geo = this->make_checked_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(real_type{0.02 * pi});

        EXPECT_FALSE(result.boundary);
        EXPECT_SOFT_EQ(real_type{0.02 * pi}, result.distance);
        EXPECT_LT(distance(Real3({std::cos(quarter), 4 + std::sin(quarter), 0}),
                           geo.pos()),
                  real_type{2e-6});
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
    real_type const circ = 2.0 * 2 * pi;

    {
        SCOPED_TRACE("Exit (twelfth of a turn)");

        auto geo = this->make_checked_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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

        auto geo = this->make_checked_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Reenter (1/3 turn)");

        auto geo = this->make_checked_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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

        auto geo = this->make_checked_track_view();
        geo.cross_boundary();
        EXPECT_EQ("inner", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Return to start (2/3 turn)");

        FieldDriverOptions driver_options;
        driver_options.max_substeps = 100;
        auto geo = this->make_checked_track_view();
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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
    real_type const circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits boundary");

        real_type dy = 1.1 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({1, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({x, 5, 0}), geo.pos()), 2e-5)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), 2e-5)
            << "Ending direction at " << geo.dir();

        if (geo.check_normal())
        {
            EXPECT_NORMAL_EQUIV((Real3{0, 1, 0}), geo.normal());
        }
        geo.cross_boundary();
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Barely misses boundary");

        real_type dy = 0.9 * driver_options.delta_chord;

        FieldDriverOptions driver_options;
        driver_options.max_substeps = 100;
        auto geo = this->make_geo_track_view({1, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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
    real_type const circ = 2 * pi;

    {
        SCOPED_TRACE("Barely hits y boundary");

        real_type dy = 1.1 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        // Trigonometry to find actual intersection point and length along arc
        real_type theta = std::asin(1 - dy);
        real_type x = std::sqrt(2 * dy - ipow<2>(dy));

        EXPECT_SOFT_NEAR(theta, result.distance, .025);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5 + x, 5, 0}), geo.pos()), 1e-5)
            << "Actually stopped at " << geo.pos();
        EXPECT_LT(distance(Real3({dy - 1, x, 0}), geo.dir()), real_type{1.5e-5})
            << "Ending direction at " << geo.dir();

        if (geo.check_normal())
        {
            EXPECT_NORMAL_EQUIV((Real3{0, 1, 0}), geo.normal());
        }

        EXPECT_NO_THROW(geo.cross_boundary());
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Hits y because the chord goes through x first");

        real_type dy = 0.001 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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

        if (geo.check_normal())
        {
            EXPECT_NORMAL_EQUIV((Real3{0, 1, 0}), geo.normal());
        }

        EXPECT_NO_THROW(geo.cross_boundary());
        EXPECT_EQ("world", this->volume_name(geo));
    }
    {
        SCOPED_TRACE("Barely (correctly) misses y");

        real_type dy = -0.001 * driver_options.delta_chord;

        auto geo = this->make_geo_track_view({-4, 4 + dy, 0}, {0, 1, 0});
        auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
            field, driver_options, particle, geo);
        auto result = propagate(circ);

        EXPECT_SOFT_NEAR(circ * .25, result.distance, 1e-5);
        EXPECT_TRUE(result.boundary);
        EXPECT_LT(distance(Real3({-5, 5 + dy, 0}), geo.pos()), 1e-5);
        EXPECT_LT(distance(Real3({-1, 0, 0}), geo.dir()), 1e-5);

        if (geo.check_normal())
        {
            EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), geo.normal());
        }
        EXPECT_NO_THROW(geo.cross_boundary());
        EXPECT_EQ("world", this->volume_name(geo));
    }
}

// Endpoint of a step is very close to the boundary.
TEST_F(TwoBoxesTest, TEST_IF_CELERITAS_DOUBLE(electron_step_endpoint))
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    UniformZField field(unit_radius_field_strength);
    FieldDriverOptions driver_options;
    real_type const dr = 0.1;
    driver_options.delta_intersection = 0.1;

    // First step length and position from starting at {0,0,0} along {0,1,0}
    static constexpr real_type first_step = 0.44815869703174;
    static constexpr Real3 first_pos
        = {-0.098753281951459, 0.43330671122068, 0};

    auto geo = this->make_checked_track_view();
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        field, particle.charge());
    auto propagate = [&](real_type start_delta, real_type move_delta) {
        Real3 start_pos{-5 + start_delta, 0, 0};
        axpy(real_type(-1), first_pos, &start_pos);

        geo = GeoTrackInitializer{start_pos, {0, 1, 0}};
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
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
        EXPECT_EQ(1, integrate.exchange_count());
        EXPECT_SOFT_EQ(first_step - dr, result.distance);
        EXPECT_LT(distance(Real3{-4.9512441890768795, -0.092139178167222446, 0},
                           geo.pos()),
                  coarse_eps)
            << geo.pos();
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
        EXPECT_EQ(1, integrate.exchange_count());
        EXPECT_SOFT_EQ(0.44815869703173999, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(-5.0, geo.pos()[0]);
        EXPECT_LT(
            distance(Real3{-4.9900002299216384, 8.2444433238682002e-08, 0},
                     geo.pos()),
            1e-6)
            << geo.pos();
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
        EXPECT_EQ(1, integrate.exchange_count());
        EXPECT_SOFT_EQ(0.44815869703173999, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(-5.0, geo.pos()[0]);
        EXPECT_LT(
            distance(Real3{-4.9999998999999997, 3.0685999199146494e-15, 0},
                     geo.pos()),
            1e-6)
            << geo.pos();
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
        EXPECT_EQ(1, integrate.exchange_count());
        EXPECT_SOFT_EQ(0.40277704609562048, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(distance(Real3{-5, -0.04387770235662955, 0}, geo.pos()),
                  coarse_eps)
            << geo.pos();
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
        EXPECT_EQ(1, integrate.exchange_count());
        EXPECT_SOFT_EQ(0.44815824321522935, result.distance);
        EXPECT_LE(result.distance, first_step);
        EXPECT_LT(distance(Real3{-5, -4.3877702173875065e-07, 0}, geo.pos()),
                  1e-6)
            << geo.pos();
    }
}

// Electron barely crosses boundary
TEST_F(TwoBoxesTest,
       TEST_IF_CELERITAS_DOUBLE(electron_tangent_cross_smallradius))
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});

    UniformZField field(unit_radius_field_strength * 100);
    real_type const radius = 0.01;
    real_type const miss_distance = 1e-4;

    std::vector<int> boundary;
    std::vector<real_type> distances;
    std::vector<int> substeps;
    std::vector<std::string> volumes;

    for (real_type dtheta :
         {pi / 4, pi / 7, real_type{1e-3}, real_type{1e-6}, real_type{1e-9}})
    {
        SCOPED_TRACE(dtheta);
        {
            // Angle of intercept with boundary
            real_type tint = std::asin((radius - miss_distance) / radius);
            real_type const sintheta = std::sin(tint - dtheta);
            real_type const costheta = std::cos(tint - dtheta);

            Real3 pos{radius * costheta,
                      5 + miss_distance - radius + radius * sintheta,
                      0};
            Real3 dir{-sintheta, costheta, 0};
            this->make_geo_track_view(pos, dir);
        }
        auto geo = this->make_checked_track_view();
        EXPECT_EQ("inner", this->volume_name(geo));

        EXPECT_SOFT_EQ(radius,
                       this->calc_field_curvature(particle, geo, field));

        // Build propagator
        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());
        FieldDriverOptions driver_options;
        driver_options.max_substeps = 100;
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        for (int i : range(2))
        {
            SCOPED_TRACE(i);
            Propagation result;
            result.distance = 0;
            result.boundary = false;

            try
            {
                if (!geo.failed())
                {
                    result = propagate(radius * dtheta);
                }
            }
            catch (CheckedGeoError const& e)
            {
                CELER_LOG(error) << e.what();
            }

            if (result.distance > 0)
            {
                boundary.push_back(result.boundary);
                volumes.push_back(this->volume_name(geo));
            }
            else
            {
                // Error sentinel
                boundary.push_back(-1);
                volumes.push_back("[FAILURE]");
            }
            distances.push_back(result.distance);
            substeps.push_back(integrate.exchange_count());
        }
    }

    std::vector<int> expected_boundary = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    std::vector<double> expected_distances = {
        0.0078534718906499,
        0.0078539816339745,
        0.0044879852658442,
        0.0044879895051283,
        1e-05,
        1e-06,
        9.9999658622419e-09,
        1e-08,
        9.9981633254417e-12,
        1e-11,
    };
    std::vector<int> expected_substeps = {1, 1, 1, 1, 1, 4, 1, 1, 1, 1};
    std::vector<std::string> expected_volumes = {
        "inner",
        "inner",
        "inner",
        "inner",
        "inner",
        "inner",
        "inner",
        "inner",
        "inner",
        "inner",
    };

    EXPECT_VEC_EQ(expected_boundary, boundary) << repr(boundary);
    EXPECT_VEC_NEAR(expected_distances, distances, real_type{.1} * coarse_eps)
        << repr(distances);
    EXPECT_VEC_EQ(expected_substeps, substeps) << repr(substeps);
    EXPECT_VEC_EQ(expected_volumes, volumes) << repr(volumes);
}

// Heuristic test: plotting points with finer propagation distance show a track
// with decreasing radius
TEST_F(TwoBoxesTest, TEST_IF_CELERITAS_DOUBLE(nonuniform_field))
{
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    ReluZField field{unit_radius_field_strength};
    FieldDriverOptions driver_options;

    auto geo = this->make_geo_track_view({-2.0, 0, 0}, {0, 1, 1});
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        field, particle.charge());
    auto propagate
        = make_field_propagator(integrate, driver_options, particle, geo);

    std::vector<real_type> all_pos;
    std::vector<int> step_counter;
    for ([[maybe_unused]] auto i : range(8))
    {
        propagate(1.0);
        all_pos.insert(all_pos.end(), geo.pos().begin(), geo.pos().end());
        step_counter.push_back(integrate.exchange_count());
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
    real_type const radius{3.8085385437789383};
    auto particle
        = this->make_particle_view(pdg::electron(), MevEnergy{10.9181415106});
    auto geo = this->make_geo_track_view({radius, 0, 0}, {0, 1, 0});
    UniformZField field(1.0_T);

    // Build propagator
    FieldDriverOptions driver_options;
    auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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
                EXPECT_REAL_EQ(expected_y[j], geo.pos()[1]);
                geo.cross_boundary();
            }
        }
    }

    EXPECT_SOFT_NEAR(-0.13150565, geo.pos()[0], coarse_eps);
    EXPECT_SOFT_NEAR(-0.03453068, geo.dir()[1], coarse_eps);
    EXPECT_SOFT_NEAR(221.48171708, total_length, coarse_eps);
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
    auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
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
            EXPECT_REAL_EQ(step, result.distance);
            ASSERT_FALSE(result.boundary);
            EXPECT_REAL_EQ(step, result.distance);
        }
    }
    EXPECT_SOFT_NEAR(2 * pi * radius * num_revs, total_length, 1e-5);
}

//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, TEST_IF_CELERITAS_DOUBLE(electron_stuck))
{
    auto particle = this->make_particle_view(pdg::electron(),
                                             MevEnergy{4.25402379798713e-01});
    UniformZField field(real_type(1_T));
    FieldDriverOptions driver_options;

    auto geo = this->make_geo_track_view(
        {-2.43293925496543e+01, -1.75522265870979e+01, 2.80918346435833e+02},
        {7.01343313647855e-01, -6.43327996599957e-01, 3.06996164784077e-01});

    auto calc_radius
        = [&geo]() { return std::hypot(geo.pos()[0], geo.pos()[1]); };
    EXPECT_SOFT_EQ(30.000000000000011, calc_radius());

    // NOTE: vecgeom 2.x-solids puts this position slightly *outside* the beam
    // tube rather than *inside*
    if (using_solids_vg && CELERITAS_VECGEOM_VERSION >= 0x020000)
    {
        // TODO: VecGeom 2.x-solids starts to diverge here
        EXPECT_EQ("vacuum_tube", this->volume_name(geo));
        GTEST_SKIP() << "FIXME: VecGeom 2.x-solid construction failure.";
    }
    EXPECT_EQ("si_tracker", this->volume_name(geo));
    {
        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        auto result = propagate(1000);
        EXPECT_EQ(result.boundary, geo.is_on_boundary());

        EXPECT_EQ("si_tracker", this->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());
        EXPECT_FALSE(result.looping);

        if (using_surface_vg)
        {
            // Surface geometry does not intersect the cylinder boundary, so
            // the track keeps going until the "looping" counter is hit
            EXPECT_SOFT_EQ(1.0314309658010318e-13, result.distance);
            EXPECT_LT(result.distance, 2e-13);
            EXPECT_FALSE(result.looping);
        }
        else
        {
            EXPECT_SOFT_EQ(29.999999999999996, calc_radius());
            EXPECT_EQ("si_tracker", this->volume_name(geo));
            ASSERT_TRUE(geo.is_on_boundary());
            if (geo.check_normal())
            {
                EXPECT_NORMAL_EQUIV(
                    (Real3{0.810979751655143, 0.58507421956993, 0}),
                    geo.normal());
            }
            geo.cross_boundary();
        }
    }
    {
        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());

        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        Propagation result;
        try
        {
            result = propagate(30);
        }
        catch (RuntimeError const& e)
        {
            FAIL() << e.what();
        }

        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_SOFT_NEAR(
            double{30}, static_cast<double>(integrate.exchange_count()), 0.2);

        if (using_surface_vg)
        {
            EXPECT_FALSE(geo.is_on_boundary());
            GTEST_SKIP() << "FIXME: VecGeom surface model fails";
        }
        ASSERT_TRUE(geo.is_on_boundary());

        if (geo.check_normal())
        {
            EXPECT_VEC_SOFT_EQ(
                (Real3{-0.819614018634831, -0.572916102459394, 0}),
                geo.normal());
        }
        EXPECT_SOFT_EQ(30, calc_radius());
        geo.cross_boundary();
        EXPECT_EQ(using_surface_vg ? "vacuum_tube" : "si_tracker",
                  this->volume_name(geo))
            << " vecgeom_version=" << std::hex << CELERITAS_VECGEOM_VERSION
            << std::dec;
    }
}

TEST_F(SimpleCmsTest, TEST_IF_CELERITAS_DOUBLE(vecgeom_failure))
{
    UniformZField field(real_type(1_T));
    FieldDriverOptions driver_options;
    driver_options.max_substeps = 100;

    // Track is really close to boundary si_tracker <- em_calorimter, at
    // at R = (125 + 3e-8) and moving almost tangentially, but able to enter
    // the si_tracker (R < 125)
    auto geo = this->make_geo_track_view({1.23254142755319734e+02,
                                          -2.08186543568394598e+01,
                                          -4.08262349901495583e+01},
                                         {-2.59700373666105766e-01,
                                          -8.11661685885768147e-01,
                                          -5.23221772848529443e-01});

    auto calc_radius
        = [&geo]() { return std::hypot(geo.pos()[0], geo.pos()[1]); };

    {
        auto particle = this->make_particle_view(
            pdg::electron(), MevEnergy{3.27089632881079409e-02});
        auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
            field, particle.charge());
        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);
        Propagation result;
        EXPECT_NO_THROW(result = propagate(1.39170198361108938e-05));
        EXPECT_EQ(result.boundary, geo.is_on_boundary());
        EXPECT_EQ("em_calorimeter", this->volume_name(geo));
        EXPECT_SOFT_EQ(125.00000000000001, calc_radius());
        EXPECT_EQ(2, integrate.exchange_count());
        EXPECT_FALSE(result.looping);
    }
    {
        ScopedLogStorer scoped_log_{&celeritas::self_logger()};
        ASSERT_TRUE(geo.is_on_boundary());
        // Simulate MSC making us reentrant
        geo.set_dir({-1.31178657592616127e-01,
                     -8.29310561920304168e-01,
                     -5.43172303859124073e-01});
        try
        {
            geo.cross_boundary();
        }
        catch (CheckedGeoError const& e)
        {
            FAIL() << e.what();
        }

        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // ORANGE should successfully reenter. However, under certain
            // system configurations, VecGeom will end up in the world volume,
            // so we don't test in all cases.
            EXPECT_EQ("em_calorimeter", this->volume_name(geo));
        }
        else
        {
            // FIXME: see GeoTests: TwoBoxesGeoTest::test_tangent
            GTEST_SKIP();
        }
    }
}

//! Get detailed results from field propagation with multiple radii
struct FieldPropagationResult
{
    std::vector<int> num_boundary;
    std::vector<int> num_step;
    std::vector<int> num_intercept;
    std::vector<int> num_integration;
    std::vector<std::vector<std::string>> messages;

    // Add a failure sentinel at a certain index
    void fail_at(std::size_t index);
};

void FieldPropagationResult::fail_at(std::size_t index)
{
    CELER_EXPECT(index < num_boundary.size());
    num_boundary[index] = -1;
    num_step[index] = -1;
    num_intercept[index] = -1;
    num_integration[index] = -1;
}

std::ostream& operator<<(std::ostream& os, FieldPropagationResult const& ref)
{
    // clang-format off
    os << "/*** FIELD PROPAGATION RESULT ***/\n"
          "FieldPropagationResult ref;\n"
       << CELER_REF_ATTR(num_boundary)
       << CELER_REF_ATTR(num_step)
       << CELER_REF_ATTR(num_intercept)
       << CELER_REF_ATTR(num_integration)
       << CELER_REF_ATTR(messages)
       << "EXPECT_REF_EQ(ref, result) << result;\n"
          "/*** END CODE ***/\n";
    // clang-format on
    return os;
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   FieldPropagationResult const& val1,
                                   FieldPropagationResult const& val2)
{
    AssertionHelper result{expr1, expr2};

#define IRE_COMPARE(ATTR)                                          \
    if (val1.ATTR != val2.ATTR)                                    \
    {                                                              \
        result.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR);           \
    }                                                              \
    else                                                           \
        CELER_DISCARD(int)

    IRE_COMPARE(num_boundary);
    IRE_COMPARE(num_step);
    IRE_COMPARE(num_intercept);
    IRE_COMPARE(num_integration);
    IRE_COMPARE(messages);

#undef IRE_COMPARE
    return result;
}

TEST_F(CmseTest, coarse)
{
    // Build propagator
    UniformZField field{0};
    auto particle = this->make_particle_view(pdg::electron(), MevEnergy{10});
    auto integrate = make_mag_field_integrator<DiagnosticDPIntegrator>(
        field, particle.charge());

    FieldDriverOptions driver_options;
    driver_options.delta_intersection = 0.001;
    driver_options.delta_chord = 0.1;
    driver_options.max_substeps = 100;

    FieldPropagationResult result;

    for (real_type radius : {5, 10, 20, 50})
    {
        ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                    LogLevel::warning};
        auto geo = this->make_geo_track_view(
            {2 * radius + real_type{0.01}, 0, -300}, {0, 1, 1});
        // TODO: define a "reentrant" different propagation status: see
        // CheckedGeoTrackView, OrangeTrackView
        geo.check_zero_distance(false);

        field = UniformZField(unit_radius_field_strength / radius);
        EXPECT_SOFT_EQ(radius,
                       this->calc_field_curvature(particle, geo, field));

        auto propagate
            = make_field_propagator(integrate, driver_options, particle, geo);

        int step_count = 0;
        int boundary_count = 0;
        constexpr int max_steps = 10000;
        while (!geo.is_outside() && !geo.failed() && step_count++ < max_steps)
        {
            Propagation result;
            try
            {
                result = propagate(radius);
                if (result.boundary)
                {
                    ++boundary_count;
                    geo.cross_boundary();
                }
            }
            catch (CheckedGeoError const& e)
            {
                CELER_LOG_LOCAL(error) << e.details().what;
                break;
            }
        }
        result.num_boundary.push_back(boundary_count);
        result.num_step.push_back(step_count);
        result.num_intercept.push_back(geo.intersect_count());
        result.num_integration.push_back(integrate.exchange_count());
        result.messages.push_back(std::move(scoped_log_).messages());
        if (geo.failed())
        {
            CELER_LOG(error)
                << "Failed radius = " << radius << " after "
                << result.num_boundary.back() << " boundary crossings, "
                << result.num_step.back() << " steps, "
                << result.num_intercept.back() << " intersection calls";
            result.fail_at(result.num_boundary.size() - 1);
        }
    }

    FieldPropagationResult ref;
    ref.num_boundary = {134, 101, 60, 40};
    ref.num_step = {10001, 6462, 3236, 1303};
    ref.num_intercept = {30419, 19551, 16170, 9956};
    ref.num_integration = {80659, 58282, 41914, 26114};
    ref.messages.resize(ref.num_boundary.size());

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        ref.fail_at(1);
        if (CELERITAS_DEBUG)
        {
            ref.messages[1] = {
                R"(Calculated surface sense at position {10.32, -6.565, 796.9} already matches target sense)",
                R"(Calculated surface sense at position {10.32, -6.565, 796.9} already matches target sense)",
                R"(Calculated surface sense at position {10.32, -6.565, 796.9} already matches target sense)",
            };
        }
        ref.messages[1].insert(
            ref.messages[1].end(),
            {R"(track failed to cross local surface 91 in universe 0 at local position {10.47, -6.625, 797.1} along local direction {0.6625, -0.2470, 0.7072})",
             R"(failed during cross_boundary: at {10.47, -6.625, 797.1} [cm] along {0.6625, -0.2470, 0.7072}, [FAILED] [ON BOUNDARY] in [OUTSIDE])"});
    }
    else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
             && !CELERITAS_VECGEOM_SURFACE
             && CELERITAS_VECGEOM_VERSION < 0x020000)
    {
        // VG Solid 2
        ref.messages[1] = {
            R"(Moved internally from boundary but safety didn't increase: volume 18 from {10.32, -6.565, 796.9} [cm] to {10.32, -6.565, 796.9} [cm] (distance: 1e-4 [cm]))"};
    }
    else if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
             && !CELERITAS_VECGEOM_SURFACE)
    {
        // VG Solid 2
        ref.messages[1] = {
            R"(Moved internally from boundary but safety didn't increase: volume 18 from {10.32, -6.565, 796.9} [cm] to {10.32, -6.565, 796.9} [cm] (distance: 1e-4 [cm]))"};
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
        || CELERITAS_VECGEOM_SURFACE)
    {
        GTEST_SKIP() << "Ignore checks due to reduced-precision numerical "
                        "sensitivity";
    }

    EXPECT_REF_EQ(ref, result) << result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

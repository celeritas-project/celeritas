//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/LocalSurfaceVisitor.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/LocalSurfaceVisitor.hh"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "corecel/random/distribution/UniformBoxDistribution.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "orange/OrangeInput.hh"
#include "orange/OrangeParams.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/surf/VariantSurface.hh"

#include "LocalSurfaceVisitor.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& os, Sense s)
{
    return os << to_char(s);
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceActionTest : public OrangeGeoTestBase
{
  protected:
    void SetUp() override
    {
        UnitInput unit;
        unit.label = "dummy";
        unit.bbox = {{-3, -2, -1}, {6, 8, 10}};
        unit.surfaces = {
            PlaneX(1),
            PlaneY(2),
            PlaneZ(3),
            CCylX(5),
            CCylY(6),
            CCylZ(7),
            SphereCentered(1.0),
            CylX({1, 2, 3}, 0.5),
            CylY({1, 2, 3}, 0.6),
            CylZ({1, 2, 3}, 0.7),
            Sphere({1, 2, 3}, 1.5),
            ConeX({1, 2, 3}, 0.2),
            ConeY({1, 2, 3}, 0.4),
            ConeZ({1, 2, 3}, 0.6),
            SimpleQuadric({0, 1, 2}, {6, 7, 8}, 9),
            GeneralQuadric({0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9),
        };
        unit.volumes = {[&unit] {
            // Create a volume
            VolumeInput v;
            for (logic_int i : range(unit.surfaces.size()))
            {
                v.logic.push_back(i);
                if (i != 0)
                {
                    v.logic.push_back(logic::lor);
                }
                v.faces.push_back(LocalSurfaceId{i});
            }
            v.logic.insert(v.logic.end(), {logic::ltrue, logic::lor});
            v.bbox = {{-1, -1, -1}, {1, 1, 1}};
            v.zorder = ZOrder::media;
            return v;
        }()};

        // Construct a single dummy volume
        this->build_geometry(std::move(unit));
    }

    void fill_uniform_box(Span<Real3> pos)
    {
        auto const& bbox = this->params().bbox();
        UniformBoxDistribution<> sample_box{bbox.lower(), bbox.upper()};
        for (Real3& d : pos)
        {
            d = sample_box(rng_);
        }
    }

    void fill_isotropic(Span<Real3> dir)
    {
        IsotropicDistribution<> sample_isotropic;
        for (Real3& d : dir)
        {
            d = sample_isotropic(rng_);
        }
    }

    std::mt19937 rng_;
};

//---------------------------------------------------------------------------//
// HELPERS
//---------------------------------------------------------------------------//

struct ToString
{
    template<class S>
    std::string operator()(S&& surf) const
    {
        std::ostringstream os;
        os << surf;
        return os.str();
    }

    // Make sure this test class is being move-constructed
    ToString() = default;
    ToString(ToString const&) = delete;
    ToString(ToString&&) = default;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

// The surface action functors are equivalent to a variant visitor
TEST_F(SurfaceActionTest, variant_surface)
{
    std::vector<VariantSurface> surfaces{
        PlaneZ{3},
        CCylX{5},
        Sphere{{1, 2, 3}, 1.5},
        ConeY{{1, 2, 3}, 0.4},
        SimpleQuadric{{0, 1, 2}, {6, 7, 8}, 9},
    };
    std::vector<std::string> strings;
    for (auto const& vs : surfaces)
    {
        strings.push_back(std::visit(ToString{}, vs));
    }
    static char const* const expected_strings[] = {
        "Plane: z=3",
        "Cyl x: r=5",
        "Sphere: r=1.5 at {1,2,3}",
        "Cone y: t=0.4 at {1,2,3}",
        "SQuadric: {0,1,2} {6,7,8} 9",
    };
    EXPECT_VEC_EQ(expected_strings, strings);
}

TEST_F(SurfaceActionTest, surface_traits_visitor)
{
    // Get the surface type reported by a surface
    auto get_surface_type = [](auto surf_traits) -> SurfaceType {
        using Surface = typename decltype(surf_traits)::type;
        return Surface::surface_type();
    };

    // Check that all surface types can be visited and are consistent
    for (auto st : range(SurfaceType::size_))
    {
        if (st == SurfaceType::inv)
        {
            if (CELERITAS_DEBUG)
            {
                // TODO: see celeritas-project/celeritas#1342
                EXPECT_THROW(visit_surface_type(get_surface_type, st),
                             DebugError);
            }
            continue;
        }

        SurfaceType actual_st = visit_surface_type(get_surface_type, st);
        EXPECT_EQ(st, actual_st);
    }
}

TEST_F(SurfaceActionTest, string)
{
    // Create functor to visit the local surface
    LocalSurfaceVisitor visit(this->host_params(), SimpleUnitId{0});

    // Loop over all surfaces and apply
    std::vector<std::string> strings;
    auto num_surf
        = this->host_params().simple_units[SimpleUnitId{0}].surfaces.size();
    for (auto id : range(LocalSurfaceId{num_surf}))
    {
        strings.push_back(visit(ToString{}, id));
    }

    static char const* const expected_strings[] = {
        "Plane: x=1",
        "Plane: y=2",
        "Plane: z=3",
        "Cyl x: r=5",
        "Cyl y: r=6",
        "Cyl z: r=7",
        "Sphere: r=1",
        "Cyl x: r=0.5 at y=2, z=3",
        "Cyl y: r=0.6 at x=1, z=3",
        "Cyl z: r=0.7 at x=1, y=2",
        "Sphere: r=1.5 at {1,2,3}",
        "Cone x: t=0.2 at {1,2,3}",
        "Cone y: t=0.4 at {1,2,3}",
        "Cone z: t=0.6 at {1,2,3}",
        "SQuadric: {0,1,2} {6,7,8} 9",
        "GQuadric: {0,1,2} {3,4,5} {6,7,8} 9",
    };
    EXPECT_VEC_EQ(expected_strings, strings);
}

TEST_F(SurfaceActionTest, host_distances)
{
    auto const& host_ref = this->host_params();

    // Create states and sample uniform box, isotropic direction
    HostVal<OrangeMiniStateData> states;
    resize(&states, host_ref, 1024);
    this->fill_uniform_box(states.pos[AllItems<Real3>{}]);
    this->fill_isotropic(states.dir[AllItems<Real3>{}]);

    CalcSenseDistanceExecutor<> calc_thread{host_ref, make_ref(states)};
    for (auto tid : range(TrackSlotId{states.size()}))
    {
        calc_thread(tid);
    }

    if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
    {
        GTEST_SKIP() << "Test results are based on double-precision RNG";
    }

    auto test_threads = range(TrackSlotId{10});
    double const expected_distance[] = {
        inf,
        inf,
        inf,
        inf,
        8.6234865826355,
        8.1154296972082,
        inf,
        inf,
        inf,
        inf,
    };

    EXPECT_EQ("{- - + + - - + + + +}",
              senses_to_string(states.sense[test_threads]));
    EXPECT_VEC_SOFT_EQ(expected_distance, states.distance[test_threads]);
}

TEST_F(SurfaceActionTest, TEST_IF_CELER_DEVICE(device_distances))
{
    auto device_states = [this] {
        // Initialize on host
        HostVal<OrangeMiniStateData> host_states;
        resize(&host_states, this->host_params(), 1024);
        this->fill_uniform_box(host_states.pos[AllItems<Real3>{}]);
        this->fill_isotropic(host_states.dir[AllItems<Real3>{}]);

        // Copy starting position/direction to device
        OrangeMiniStateData<Ownership::value, MemSpace::device> device_states;
        device_states = host_states;
        return device_states;
    }();

    // Launch kernel
    SATestInput input;
    input.params = this->params().device_ref();
    input.states = device_states;
    sa_test(input);

    // Copy result back to host
    auto host_states = [&device_states] {
        HostVal<OrangeMiniStateData> host_states;
        host_states = device_states;
        return host_states;
    }();

    if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
    {
        GTEST_SKIP() << "Test results are based on double-precision RNG";
    }

    auto test_threads = range(TrackSlotId{10});
    double const expected_distance[] = {
        inf, inf, inf, inf, 8.623486582635, 8.115429697208, inf, inf, inf, inf};
    EXPECT_EQ("{- - + + - - + + + +}",
              senses_to_string(host_states.sense[test_threads]));
    EXPECT_VEC_SOFT_EQ(expected_distance, host_states.distance[test_threads]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

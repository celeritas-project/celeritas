//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceAction.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SurfaceAction.hh"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionMirror.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "orange/OrangeParams.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/construct/SurfaceInputBuilder.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/surf/Surfaces.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

#include "SurfaceAction.test.hh"
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

        {
            // Build surfaces
            SurfaceInputBuilder insert(&unit.surfaces);
            insert(PlaneX(1), "px");
            insert(PlaneY(2), "py");
            insert(PlaneZ(3), "pz");
            insert(CCylX(5), "mycylx");
            insert(CCylY(6), "mycyly");
            insert(CCylZ(7), "mycylz");
            insert(SphereCentered(1.0), "csph");
            insert(CylX({1, 2, 3}, 0.5), "acylx");
            insert(CylY({1, 2, 3}, 0.6), "acyly");
            insert(CylZ({1, 2, 3}, 0.7), "acylz");
            insert(Sphere({1, 2, 3}, 1.5), "mysph");
            insert(ConeX({1, 2, 3}, 0.2), "aconex");
            insert(ConeY({1, 2, 3}, 0.4), "aconey");
            insert(ConeZ({1, 2, 3}, 0.6), "aconez");
            insert(SimpleQuadric({0, 1, 2}, {6, 7, 8}, 9), "sq");
            insert(GeneralQuadric({0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9), "gq");
            EXPECT_EQ(16, unit.surfaces.size());
        }
        {
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
            unit.volumes = {std::move(v)};
        }
        {
            unit.bbox = {{-3, -2, -1}, {6, 8, 10}};
        }

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

class StaticSurfaceActionTest : public Test
{
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
//! Get the amount of storage
template<class S>
struct GetStorageSize
{
    constexpr size_type operator()() const noexcept
    {
        return S::Storage::extent * sizeof(typename S::Storage::value_type);
    }
};

//---------------------------------------------------------------------------//
//! Get the actual size of a surface instance
template<class S>
struct GetTypeSize
{
    constexpr size_type operator()() const noexcept { return sizeof(S); }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SurfaceActionTest, string)
{
    // Create functor
    auto const& host_ref = this->host_params();
    Surfaces surfaces(host_ref,
                      host_ref.simple_units[SimpleUnitId{0}].surfaces);
    auto surf_to_string = make_surface_action(surfaces, ToString{});

    // Loop over all surfaces and apply
    std::vector<std::string> strings;
    for (auto id : range(LocalSurfaceId{surfaces.num_surfaces()}))
    {
        strings.push_back(surf_to_string(id));
    }

    char const* const expected_strings[] = {
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
    HostRef<OrangeMiniStateData> state_ref;
    state_ref = states;
    this->fill_uniform_box(state_ref.pos[AllItems<Real3>{}]);
    this->fill_isotropic(state_ref.dir[AllItems<Real3>{}]);

    CalcSenseDistanceExecutor<> calc_thread{host_ref, state_ref};
    for (auto tid : range(TrackSlotId{states.size()}))
    {
        calc_thread(tid);
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
              senses_to_string(state_ref.sense[test_threads]));
    EXPECT_VEC_SOFT_EQ(expected_distance, state_ref.distance[test_threads]);
}

TEST_F(SurfaceActionTest, TEST_IF_CELER_DEVICE(device_distances))
{
    OrangeMiniStateData<Ownership::value, MemSpace::device> device_states;
    {
        // Initialize on host
        HostVal<OrangeMiniStateData> host_states;
        resize(&host_states, this->host_params(), 1024);
        this->fill_uniform_box(host_states.pos[AllItems<Real3>{}]);
        this->fill_isotropic(host_states.dir[AllItems<Real3>{}]);

        // Copy starting position/direction to device
        device_states = host_states;
    }

    // Launch kernel
    SATestInput input;
    input.params = this->params().device_ref();
    input.states = device_states;
    sa_test(input);

    {
        // Copy result back to host
        HostVal<OrangeMiniStateData> host_states;
        host_states = device_states;
        auto test_threads = range(TrackSlotId{10});

        double const expected_distance[] = {inf,
                                            inf,
                                            inf,
                                            inf,
                                            8.623486582635,
                                            8.115429697208,
                                            inf,
                                            inf,
                                            inf,
                                            inf};
        EXPECT_EQ("{- - + + - - + + + +}",
                  senses_to_string(host_states.sense[test_threads]));
        EXPECT_VEC_SOFT_EQ(expected_distance,
                           host_states.distance[test_threads]);
    }
}

//---------------------------------------------------------------------------//
//! Loop through all surface types and ensure "storage" type is correctly sized
TEST_F(StaticSurfaceActionTest, check_surface_sizes)
{
    auto get_expected_storage = make_static_surface_action<GetTypeSize>();
    auto get_actual_storage = make_static_surface_action<GetStorageSize>();

    for (auto st : range(SurfaceType::size_))
    {
        EXPECT_EQ(get_expected_storage(st), get_actual_storage(st));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

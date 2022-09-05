//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
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
#include "orange/Data.hh"
#include "orange/construct/SurfaceInserter.hh"
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

namespace
{
// Disabled since it's unused; it could be in the future though.
#if 0
std::vector<Sense> string_to_senses(std::string s)
{
    std::vector<Sense> result(s.size());
    std::transform(s.begin(), s.end(), result.begin(), [](char c) {
        CELER_EXPECT(c == '+' || c == '-');
        return c == '+' ? Sense::outside : Sense::inside;
    });
    return result;
}
#endif

std::string senses_to_string(Span<const Sense> s)
{
    std::string result(s.size(), ' ');
    std::transform(s.begin(), s.end(), result.begin(), [](Sense s) {
        return to_char(s);
    });
    return result;
}
} // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceActionTest : public Test
{
  protected:
    using SurfaceDataMirror = CollectionMirror<SurfaceData>;

    void SetUp() override
    {
        HostVal<SurfaceData> surface_data;
        SurfaceInserter      insert(&surface_data);
        insert(PlaneX(1));
        insert(PlaneY(2));
        insert(PlaneZ(3));
        insert(CCylX(5));
        insert(CCylY(6));
        insert(CCylZ(7));
        insert(Sphere({1, 2, 3}, 1.5));
        insert(GeneralQuadric({0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9));

        surf_params_ = SurfaceDataMirror{std::move(surface_data)};
    }

    void fill_uniform_box(Span<Real3> pos)
    {
        UniformBoxDistribution<> sample_box{{-3, -2, -1}, {6, 8, 10}};
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

    SurfaceDataMirror surf_params_;
    std::mt19937      rng_;
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
    ToString()                = default;
    ToString(const ToString&) = delete;
    ToString(ToString&&)      = default;
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
    Surfaces surfaces(surf_params_.host());
    auto     surf_to_string = make_surface_action(surfaces, ToString{});

    // Loop over all surfaces and apply
    std::vector<std::string> strings;
    for (auto id : range(SurfaceId{surfaces.num_surfaces()}))
    {
        strings.push_back(surf_to_string(id));
    }

    // clang-format off
    const std::string expected_strings[] = {
        "Plane: x=1",
        "Plane: y=2",
        "Plane: z=3",
        "Cyl x: r=5",
        "Cyl y: r=6",
        "Cyl z: r=7",
        "Sphere: r=1.5 at {1,2,3}",
        "GQuadric: {0,1,2} {3,4,5} {6,7,8} 9"};
    // clang-format on
    EXPECT_VEC_EQ(expected_strings, strings);
}

TEST_F(SurfaceActionTest, host_distances)
{
    // Create states and sample uniform box, isotropic direction
    HostVal<OrangeMiniStateData> states;
    resize(&states, surf_params_.host(), 1024);
    HostRef<OrangeMiniStateData> state_ref;
    state_ref = states;
    this->fill_uniform_box(state_ref.pos[AllItems<Real3>{}]);
    this->fill_isotropic(state_ref.dir[AllItems<Real3>{}]);

    CalcSenseDistanceLauncher<> calc_thread{surf_params_.host(), state_ref};
    for (auto tid : range(ThreadId{states.size()}))
    {
        calc_thread(tid);
    }

    auto test_threads = range(ThreadId{10});
    // PRINT_EXPECTED(senses_to_string(state_ref.sense[test_threads]));
    // PRINT_EXPECTED(state_ref.distance[test_threads]);

    const char expected_senses[]
        = {'-', '-', '+', '+', '-', '-', '+', '+', '-', '-'};
    const double expected_distance[] = {inf,
                                        inf,
                                        inf,
                                        inf,
                                        8.623486582635,
                                        8.115429697208,
                                        inf,
                                        5.436749550654,
                                        0.9761596300109,
                                        5.848454015622};

    EXPECT_VEC_EQ(expected_senses,
                  senses_to_string(state_ref.sense[test_threads]));
    EXPECT_VEC_SOFT_EQ(expected_distance, state_ref.distance[test_threads]);
}

TEST_F(SurfaceActionTest, TEST_IF_CELER_DEVICE(device_distances))
{
    OrangeMiniStateData<Ownership::value, MemSpace::device> device_states;
    {
        // Initialize on host
        HostVal<OrangeMiniStateData> host_states;
        resize(&host_states, surf_params_.host(), 1024);
        this->fill_uniform_box(host_states.pos[AllItems<Real3>{}]);
        this->fill_isotropic(host_states.dir[AllItems<Real3>{}]);

        // Copy starting position/direction to device
        device_states = host_states;
    }

    // Launch kernel
    SATestInput input;
    input.params = surf_params_.device();
    input.states = device_states;
    sa_test(input);

    {
        // Copy result back to host
        HostVal<OrangeMiniStateData> host_states;
        host_states       = device_states;
        auto test_threads = range(ThreadId{10});

        const char expected_senses[]
            = {'-', '-', '+', '+', '-', '-', '+', '+', '-', '-'};
        const double expected_distance[] = {inf,
                                            inf,
                                            inf,
                                            inf,
                                            8.623486582635,
                                            8.115429697208,
                                            inf,
                                            5.436749550654,
                                            0.9761596300109,
                                            5.848454015622};
        EXPECT_VEC_EQ(expected_senses,
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
    auto get_actual_storage   = make_static_surface_action<GetStorageSize>();

    for (auto st : range(SurfaceType::size_))
    {
        EXPECT_EQ(get_expected_storage(st), get_actual_storage(st));
    }
}
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas

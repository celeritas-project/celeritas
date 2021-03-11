//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Collection.test.cc
//---------------------------------------------------------------------------//
#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/CollectionMirror.hh"

#include <cstdint>
#include <type_traits>
#include "celeritas_test.hh"
#include "Collection.test.hh"
#include "base/DeviceVector.hh"
#include "comm/Device.hh"

using celeritas::Collection;
using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::Span;
using celeritas::ThreadId;

using namespace celeritas_test;

template<class T>
constexpr bool is_trivial_v = std::is_trivially_copyable<T>::value;

TEST(SimpleCollection, range_types)
{
    EXPECT_TRUE((is_trivial_v<celeritas::ItemRange<int>>));
    EXPECT_TRUE(
        (is_trivial_v<
            celeritas::Collection<int, Ownership::reference, MemSpace::device>>));
    EXPECT_TRUE((is_trivial_v<celeritas::Collection<int,
                                                    Ownership::const_reference,
                                                    MemSpace::device>>));
}

// NOTE: these tests are essentially redundant with Range.test.cc since
// ItemRange is a Range<OpaqueId> and ItemId is an OpaqueId.
TEST(SimpleCollection, range)
{
    using ItemRangeT = celeritas::ItemRange<int>;
    using ItemIdT    = celeritas::ItemId<int>;
    ItemRangeT ps;
    EXPECT_EQ(0, ps.size());
    EXPECT_TRUE(ps.empty());

    ps = ItemRangeT{ItemIdT{10}, ItemIdT{21}};
    EXPECT_FALSE(ps.empty());
    EXPECT_EQ(11, ps.size());
    EXPECT_EQ(10, ps.begin()->unchecked_get());
    EXPECT_EQ(21, ps.end()->unchecked_get());

    EXPECT_EQ(ItemIdT{10}, ps[0]);
    EXPECT_EQ(ItemIdT{12}, ps[2]);
}

TEST(SimpleCollection, size_limits)
{
    using IdType = celeritas::OpaqueId<struct Tiny, std::uint8_t>;
    Collection<double, Ownership::value, MemSpace::host, IdType> host_val;
    auto                build = make_builder(&host_val);
    std::vector<double> dummy(254);
    auto                irange = build.insert_back(dummy.begin(), dummy.end());
    EXPECT_EQ(0, irange.begin()->unchecked_get());
    EXPECT_EQ(254, irange.end()->unchecked_get());

    // Inserting a 255-element "range" would have caused an exception in debug
    // because the "final" value `uint8_t(-1) = 255` of OpaqueId is
    // reserved. Let's say that inserting N-1 elements is "unspecified"
    // behavior -- but for now it should be OK to insert 255 as long as it's
    // with a push_back and not a range insertion.
    build.push_back(123);

#if CELERITAS_DEBUG
    // Inserting 256 elements when 255 is the max int *must* raise an error
    // when debug assertions are enabled.
    EXPECT_THROW(build.push_back(1234.5), celeritas::DebugError);
#else
    // With bounds checking disabled, a one-off check when getting a reference
    // should catch the size failure.
    build.push_back(12345.6);
    Collection<double, Ownership::const_reference, MemSpace::host, IdType> host_ref;
    EXPECT_THROW(host_ref = host_val, celeritas::RuntimeError);
#endif
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CollectionTest : public celeritas::Test
{
  protected:
    using MockParamsMirror = celeritas::CollectionMirror<MockParamsData>;

    void SetUp() override
    {
        MockParamsData<Ownership::value, MemSpace::host> host_data;
        host_data.max_element_components = 3;

        auto el_builder  = make_builder(&host_data.elements);
        auto mat_builder = make_builder(&host_data.materials);
        el_builder.reserve(5);
        mat_builder.reserve(2);

        //// Construct materials and elements ////
        {
            MockMaterial m;
            m.number_density             = 2.0;
            const MockElement elements[] = {
                {1, 1.1},
                {3, 5.0},
                {6, 12.0},
            };
            m.elements = el_builder.insert_back(std::begin(elements),
                                                std::end(elements));
            EXPECT_EQ(3, m.elements.size());
            auto id = mat_builder.push_back(m);
            EXPECT_EQ(MockMaterialId{0}, id);
        }
        {
            MockMaterial m;
            m.number_density = 20.0;
            m.elements       = el_builder.insert_back({{10, 20.0}});
            mat_builder.push_back(m);
        }
        {
            MockMaterial m;
            m.number_density = 0.0;
            mat_builder.push_back(m);
        }
        EXPECT_EQ(3, mat_builder.size());
        EXPECT_EQ(3, host_data.materials.size());
        EXPECT_EQ(4, host_data.elements.size());

        // Test host-accessible values and const correctness
        {
            const auto& host_data_const = host_data;

            const MockMaterial& m
                = host_data_const.materials[MockMaterialId{0}];
            EXPECT_EQ(3, m.elements.size());
            Span<const MockElement> els = host_data_const.elements[m.elements];
            EXPECT_EQ(3, els.size());
            EXPECT_EQ(6, els[2].atomic_number);
        }

        // Create references and copy to device if enabled
        mock_params = MockParamsMirror{std::move(host_data)};
    }

    MockParamsMirror mock_params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CollectionTest, host)
{
    MockStateData<Ownership::value, MemSpace::host>     host_state;
    MockStateData<Ownership::reference, MemSpace::host> host_state_ref;

    make_builder(&host_state.matid).resize(1);
    host_state_ref = host_state;

    // Assign
    host_state_ref.matid[ThreadId{0}] = MockMaterialId{1};

    // Create view
    MockTrackView mock(
        mock_params.host(), host_state_ref, celeritas::ThreadId{0});
    EXPECT_EQ(1, mock.matid().unchecked_get());
}

TEST_F(CollectionTest, TEST_IF_CELERITAS_CUDA(device))
{
    // Construct with 1024 states
    MockStateData<Ownership::value, MemSpace::device> device_states;
    make_builder(&device_states.matid).resize(1024);

    celeritas::DeviceVector<double> device_result(device_states.size());

    CTestInput kernel_input;
    kernel_input.params = this->mock_params.device();
    kernel_input.states = device_states;
    kernel_input.result = device_result.device_pointers();

    col_cuda_test(kernel_input);
    std::vector<double> result(device_result.size());
    device_result.copy_to_host(celeritas::make_span(result));

    // For brevity, only check the first 6 values (they repeat after that)
    result.resize(6);
    const double expected_result[] = {2.2, 41, 0, 3.333333333333, 41, 0};
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

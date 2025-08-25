//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Collection.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Collection.hh"

#include <cstdint>
#include <random>
#include <type_traits>

#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"

#include "Collection.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

template<class T>
constexpr bool is_trivial_v = std::is_trivially_copyable<T>::value;

TEST(ItemRange, types)
{
    EXPECT_TRUE((is_trivial_v<ItemRange<int>>));
    EXPECT_TRUE((
        is_trivial_v<Collection<int, Ownership::reference, MemSpace::device>>));
    EXPECT_TRUE(
        (is_trivial_v<
            Collection<int, Ownership::const_reference, MemSpace::device>>));
}

// NOTE: these tests are essentially redundant with Range.test.cc since
// ItemRange is a Range<OpaqueId> and ItemId is an OpaqueId.
TEST(ItemRange, accessors)
{
    using ItemRangeT = ItemRange<int>;
    using ItemIdT = ItemId<int>;
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

TEST(ItemMap, basic)
{
    using T1 = OpaqueId<double>;
    using T2 = OpaqueId<int>;
    using ItemMap = ItemMap<T1, T2>;

    Collection<double, Ownership::value, MemSpace::host, T2> host_val;
    std::vector<int> data_a = {5, 6, 7, 8};
    std::vector<int> data_b = {9, 10, 11};

    // Add both vectors to the Collection, creating a Range for each
    auto range_a
        = make_builder(&host_val).insert_back(data_a.begin(), data_a.end());
    auto range_b
        = make_builder(&host_val).insert_back(data_b.begin(), data_b.end());

    ItemMap im_a;
    ItemMap im_b;

    EXPECT_EQ(0, im_a.size());
    EXPECT_TRUE(im_a.empty());

    // Create an ItemMap for each Range
    im_a = ItemMap(range_a);
    im_b = ItemMap(range_b);

    EXPECT_EQ(4, im_a.size());
    EXPECT_EQ(3, im_b.size());

    EXPECT_FALSE(im_a.empty());
    EXPECT_FALSE(im_b.empty());

    // Verify we can access the T2 data with index type T1
    for (size_type i : range(data_a.size()))
    {
        EXPECT_EQ(range_a[i], im_a[T1{i}]);
    }

    for (size_type i : range(data_b.size()))
    {
        EXPECT_EQ(range_b[i], im_b[T1{i}]);
    }
}

TEST(CollectionBuilder, accessors)
{
    Collection<int, Ownership::value, MemSpace::host> data;
    using IdType = OpaqueId<int>;
    CollectionBuilder builder{&data};
    EXPECT_EQ(0, builder.size());
    EXPECT_EQ(IdType{0}, builder.size_id());
}

TEST(CollectionBuilder, size_limits)
{
    using IdType = OpaqueId<struct Tiny_, std::uint8_t>;
    Collection<double, Ownership::value, MemSpace::host, IdType> host_val;
    auto build = CollectionBuilder(&host_val);

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(build.reserve(257), DebugError);
        EXPECT_THROW(build.resize(1000), DebugError);
    }

    std::vector<double> dummy(254);
    auto irange = build.insert_back(dummy.begin(), dummy.end());
    EXPECT_EQ(0, irange.begin()->unchecked_get());
    EXPECT_EQ(254, irange.end()->unchecked_get());

    // Inserting a 255-element "range" would have caused an exception in debug
    // because the "final" value `uint8_t(-1) = 255` of OpaqueId is
    // reserved. Let's say that inserting N-1 elements is "unspecified"
    // behavior -- but for now it should be OK to insert 255 as long as it's
    // with a push_back and not a range insertion.
    build.push_back(123);

    if (CELERITAS_DEBUG)
    {
        // Inserting 256 elements when 255 is the max int *must* raise an error
        // when debug assertions are enabled.
        EXPECT_THROW(build.push_back(1234.5), DebugError);
    }
    else
    {
        // With bounds checking disabled, a one-off check when getting a
        // reference should catch the size failure.
        build.push_back(12345.6);
        Collection<double, Ownership::const_reference, MemSpace::host, IdType>
            host_ref;
        EXPECT_THROW(host_ref = host_val, RuntimeError);
    }
}

TEST(DedupeCollectionBuilder, construction)
{
    Collection<int, Ownership::value, MemSpace::host> host_val;
    using Id = decltype(host_val)::ItemIdT;

    DedupeCollectionBuilder ints{&host_val};
    EXPECT_EQ(0, ints.size());
    EXPECT_EQ(Id{0}, ints.size_id());

    // Reserve space
    ints.reserve(10);

    auto r = ints.insert_back({1, 2, 3});
    EXPECT_EQ(0, r.begin()->unchecked_get());
    EXPECT_EQ(3, r.end()->unchecked_get());

    r = ints.insert_back({5, 4});
    EXPECT_EQ(3, r.begin()->unchecked_get());
    EXPECT_EQ(5, r.end()->unchecked_get());

    // NOTE: Sub-ranges don't get deduplicated
    r = ints.insert_back({2, 3});
    EXPECT_EQ(5, r.begin()->unchecked_get());
    EXPECT_EQ(7, r.end()->unchecked_get());

    // Test duplicate insertion
    r = ints.insert_back({5, 4});
    EXPECT_EQ(3, r.begin()->unchecked_get());
    EXPECT_EQ(5, r.end()->unchecked_get());
    r = [&ints] {
        // Different type but gets converted
        std::vector<unsigned int> temp{1, 2, 3};
        return ints.insert_back(temp.begin(), temp.end());
    }();
    EXPECT_EQ(0, r.begin()->unchecked_get());
    EXPECT_EQ(3, r.end()->unchecked_get());

    // Single-element pushes don't get deduplicated
    EXPECT_EQ(7, ints.push_back(1).unchecked_get());

    static int const expected[] = {1, 2, 3, 5, 4, 2, 3, 1};
    EXPECT_VEC_EQ(expected, host_val[AllItems<int>{}]);
}

TEST(DedupeCollectionBuilder, double_stress)
{
    using CollectionT = Collection<double, Ownership::value, MemSpace::host>;
    using RangeT = CollectionT::ItemRangeT;

    CollectionT host_val;
    constexpr size_type chunks_per_test = 8000;
    constexpr size_type items_per_chunk = 8;

    DedupeCollectionBuilder dbls{&host_val};
    std::vector<RangeT> all_inserted;
    Array<double, items_per_chunk> buffer;

    all_inserted.reserve(chunks_per_test);
    dbls.reserve(chunks_per_test * items_per_chunk);

    std::mt19937 rng;
    std::uniform_real_distribution<double> sample_urd{-1.0, 1.0};
    auto sample_uniform = [&rng, &sample_urd] { return sample_urd(rng); };

    // Insert a bunch of independent chunks
    for (auto i : range(chunks_per_test))
    {
        // Fill buffer
        std::generate(buffer.begin(), buffer.end(), sample_uniform);

        auto inserted = dbls.insert_back(buffer.begin(), buffer.end());
        EXPECT_EQ(i * items_per_chunk, inserted.begin()->unchecked_get());
        all_inserted.push_back(inserted);
    }

    ASSERT_EQ(chunks_per_test * items_per_chunk, host_val.size());
    ASSERT_EQ(chunks_per_test, all_inserted.size());

    // Reorder inserted ranges
    std::shuffle(all_inserted.begin(), all_inserted.end(), rng);

    // Loop over all previously inserted data and re-insert
    for (auto r : all_inserted)
    {
        // Get a span to the previously inserted range
        auto s = host_val[r];

        // Re-insert using dedupe: should result in same range
        auto new_r = dbls.insert_back(s.begin(), s.end());
        EXPECT_EQ(r.begin()->unchecked_get(), new_r.begin()->unchecked_get());
        EXPECT_EQ(r.end()->unchecked_get(), new_r.end()->unchecked_get());
    }
}

//---------------------------------------------------------------------------//
// SIMPLE TESTS
//---------------------------------------------------------------------------//

class SimpleCollectionTest : public Test
{
  protected:
    using IntId = ItemId<int>;
    using IntRange = ItemRange<int>;
    template<MemSpace M>
    using AllInts = AllItems<int, M>;

    template<MemSpace M>
    using Value = Collection<int, Ownership::value, M>;
    template<MemSpace M>
    using Ref = Collection<int, Ownership::reference, M>;
    template<MemSpace M>
    using CRef = Collection<int, Ownership::const_reference, M>;

    static constexpr MemSpace host = MemSpace::host;
    static constexpr MemSpace device = MemSpace::device;
};

TEST_F(SimpleCollectionTest, accessors)
{
    Value<host> host_val;
    EXPECT_TRUE(host_val.empty());

    auto irange = make_builder(&host_val).insert_back({0, 1, 2, 3});
    EXPECT_EQ(4, host_val.size());
    EXPECT_FALSE(host_val.empty());

    EXPECT_EQ(2, host_val[IntId{2}]);
    EXPECT_EQ(4, host_val[irange].size());
    EXPECT_EQ(4, host_val[AllInts<host>{}].size());

    Ref<host> host_ref(host_val);
    EXPECT_EQ(4, host_ref.size());
    host_ref = {};
    EXPECT_EQ(0, host_ref.size());
    host_ref = host_val;
    EXPECT_EQ(4, host_ref.size());

    host_ref[IntId{0}] = 123;
    EXPECT_EQ(123, host_val[IntId{0}]);
    EXPECT_EQ(123, host_ref[IntId{0}]);
    host_ref[irange].back() = 321;
    EXPECT_EQ(321, host_ref[IntId{3}]);

    Ref<host> const& host_ref_cref = host_ref;
    EXPECT_TRUE((std::is_same_v<decltype(host_ref_cref[IntId{0}]), int&>));
    EXPECT_TRUE((std::is_same_v<decltype(host_ref_cref[irange]), Span<int>>));
    EXPECT_TRUE(
        (std::is_same_v<decltype(host_ref_cref[AllInts<host>{}]), Span<int>>));
    EXPECT_EQ(123, host_ref_cref[IntId{0}]);
    EXPECT_EQ(321, host_ref_cref[irange].back());
    EXPECT_EQ(321, host_ref_cref[AllInts<host>{}].back());

    CRef<host> host_cref{host_val};
    EXPECT_TRUE((std::is_same_v<decltype(host_cref[IntId{0}]), int const&>));
    EXPECT_TRUE((std::is_same_v<decltype(host_cref[irange]), Span<int const>>));
    EXPECT_TRUE((
        std::is_same_v<decltype(host_cref[AllInts<host>{}]), Span<int const>>));
    EXPECT_EQ(4, host_ref.size());
    EXPECT_EQ(123, host_cref[IntId{0}]);
    EXPECT_EQ(123, host_cref[irange].front());
    EXPECT_EQ(321, host_cref[AllInts<host>{}].back());
}

TEST_F(SimpleCollectionTest, algo_host)
{
    Value<host> src;

    // Test 'fill'
    resize(&src, 4);
    fill(123, &src);
    EXPECT_EQ(123, src[IntId{0}]);
    EXPECT_EQ(123, src[IntId{3}]);

    // Test 'fill_sequence'
    fill_sequence(&src, StreamId{});
    for (size_type i = 0; i < src.size(); ++i)
    {
        EXPECT_EQ(i, src[IntId{i}]);
    }

    // Test 'copy_to_host'
    std::vector<int> dst(src.size());
    copy_to_host(src, make_span(dst));
    for (size_type i = 0; i < src.size(); ++i)
    {
        EXPECT_EQ(i, dst[i]);
    }
}

TEST_F(SimpleCollectionTest, TEST_IF_CELER_DEVICE(algo_device))
{
    Value<device> src;
    resize(&src, 2);
    fill(123, &src);

    CRef<device> device_cref{src};
    EXPECT_TRUE((std::is_same_v<decltype(device_cref[IntId{0}]), int>));
    EXPECT_TRUE(
        (std::is_same_v<decltype(device_cref[IntRange{IntId{0}, IntId{2}}]),
                        LdgSpan<int const>>));
    EXPECT_TRUE((std::is_same_v<decltype(device_cref[AllInts<device>{}]),
                                LdgSpan<int const>>));

    // Test 'fill_sequence'
    fill_sequence(&src, StreamId{});

    // Test 'copy_to_host'
    std::vector<int> dst(src.size());
    copy_to_host(src, make_span(dst));
    for (size_type i = 0; i < src.size(); ++i)
    {
        EXPECT_EQ(i, dst[i]);
    }
}

//---------------------------------------------------------------------------//
// ASSIGNMENT TESTS
//---------------------------------------------------------------------------//

class AssignmentTest : public SimpleCollectionTest
{
  protected:
    void SetUp()
    {
        CollectionBuilder{&host_val_}.insert_back({0, 1, 2, 3});
        ASSERT_EQ(4, host_val_.size());
        host_ref_ = host_val_;
        ASSERT_EQ(4, host_ref_.size());
        host_cref_ = host_val_;
        ASSERT_EQ(4, host_cref_.size());
    }

    Value<host> host_val_;
    Ref<host> host_ref_;
    CRef<host> host_cref_;
};

TEST_F(AssignmentTest, host_host)
{
    {
        // Assignment: ref -> value
        Value<host> temp;
        temp = host_ref_;
        EXPECT_EQ(4, temp.size());
    }
    {
        // Assignment: cref -> value
        Value<host> temp;
        temp = host_cref_;
        EXPECT_EQ(4, temp.size());
    }
    {
        // Assignment: value -> value
        Value<host> temp;
        temp = host_val_;
        EXPECT_EQ(4, temp.size());
    }
    if constexpr (false)
    {
        // PROHIBITED: cref -> ref
        Ref<host> temp;
        temp = host_cref_;
        EXPECT_EQ(4, temp.size());
    }
}

TEST_F(AssignmentTest, TEST_IF_CELER_DEVICE(host_device))
{
    {
        // Assignment: ref -> value
        Value<device> temp;
        temp = host_ref_;
        EXPECT_EQ(4, temp.size());
    }
    {
        // Assignment: cref -> value
        Value<device> temp;
        temp = host_cref_;
        EXPECT_EQ(4, temp.size());
    }
    {
        // Assignment: value -> value
        Value<device> temp;
        temp = host_val_;
        EXPECT_EQ(4, temp.size());
    }
}

TEST_F(AssignmentTest, TEST_IF_CELER_DEVICE(device_host))
{
    Value<device> device_val;
    Ref<device> device_ref;
    CRef<device> device_cref;

    device_val = host_val_;
    ASSERT_EQ(4, device_val.size());
    device_ref = device_val;
    ASSERT_EQ(4, device_ref.size());
    device_cref = device_val;
    ASSERT_EQ(4, device_cref.size());

    {
        // Assignment from value
        Value<host> temp;
        temp = device_val;
        EXPECT_EQ(4, temp.size());

        Ref<host> temp_ref;
        // First copy to incorrectly sized vector
        EXPECT_THROW(temp_ref = device_val, RuntimeError);

        // Now copy correct size
        temp_ref = temp;
        EXPECT_EQ(1, temp[ItemId<int>{1}]);
        temp[ItemId<int>{2}] = -1;
        temp_ref = device_val;
        EXPECT_EQ(2, temp[ItemId<int>{2}]);
    }
    {
        // Assignment from ref
        Value<host> temp;
        resize(&temp, 4);
        EXPECT_EQ(4, temp.size());

        Ref<host> temp_ref{temp};
        temp_ref = device_val;
        EXPECT_EQ(2, temp[ItemId<int>{2}]);
    }
}

//---------------------------------------------------------------------------//
// COMPLEX TESTS
//---------------------------------------------------------------------------//

class CollectionTest : public Test
{
  protected:
    using MockParamsMirror = CollectionMirror<MockParamsData>;

    void SetUp() override
    {
        HostVal<MockParamsData> host_data;
        host_data.max_element_components = 3;

        auto el_builder = make_builder(&host_data.elements);
        auto mat_builder = make_builder(&host_data.materials);
        el_builder.reserve(5);
        mat_builder.reserve(2);

        //// Construct materials and elements ////
        {
            MockMaterial m;
            m.number_density = 2.0;
            MockElement const elements[] = {
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
            m.elements = el_builder.insert_back({{10, 20.0}});
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
            auto const& host_data_const = host_data;

            MockMaterial const& m
                = host_data_const.materials[MockMaterialId{0}];
            EXPECT_EQ(3, m.elements.size());
            Span<MockElement const> els = host_data_const.elements[m.elements];
            EXPECT_EQ(3, els.size());
            EXPECT_EQ(6, els[2].atomic_number);
        }

        // Test references helpers
        {
            auto host_cref = make_const_ref(host_data);
            EXPECT_TRUE((std::is_same<decltype(host_cref),
                                      MockParamsData<Ownership::const_reference,
                                                     MemSpace::host>>::value));

            auto const& host_value_const = host_data;
            auto host_cref2 = make_const_ref(host_value_const);
            EXPECT_TRUE((std::is_same<decltype(host_cref2),
                                      MockParamsData<Ownership::const_reference,
                                                     MemSpace::host>>::value));
        }

        // Create references and copy to device if enabled
        mock_params = MockParamsMirror{std::move(host_data)};
    }

    MockParamsMirror mock_params;
};

template<MemSpace M>
inline void resize(MockStateData<Ownership::value, M>* data, size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&data->matid, size);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CollectionTest, host)
{
    HostVal<MockStateData> host_state;
    resize(&host_state, 1);
    auto host_state_ref = make_ref(host_state);
    host_state_ref.matid[TrackSlotId{0}] = MockMaterialId{1};

    // Create view
    MockTrackView mock(mock_params.host_ref(), host_state_ref, TrackSlotId{0});
    EXPECT_EQ(1, mock.matid().unchecked_get());
}

TEST_F(CollectionTest, TEST_IF_CELER_DEVICE(device))
{
    // Construct with 1024 states
    MockStateData<Ownership::value, MemSpace::device> device_states;
    resize(&device_states, 1024);

    DeviceVector<double> device_result(device_states.size());

    CTestInput kernel_input;
    kernel_input.params = this->mock_params.device_ref();
    kernel_input.states = device_states;
    kernel_input.result = device_result.device_ref();

    col_cuda_test(kernel_input);
    std::vector<double> result(device_result.size());
    device_result.copy_to_host(make_span(result));

    // For brevity, only check the first 6 values (they repeat after that)
    result.resize(6);
    static double const expected_result[] = {2.2, 41, 0, 3.333333333333, 41, 0};
    EXPECT_VEC_SOFT_EQ(expected_result, result);

    {
        // Create an in-memory copy of the entire collection group
        auto host_states = make_host_val(device_states);
        EXPECT_EQ(1024, host_states.size());

        // Create a track view
        auto host_ref = make_ref(host_states);
        MockTrackView mtv{mock_params.host_ref(), host_ref, TrackSlotId{2}};
        EXPECT_EQ(MockMaterialId{2}, mtv.matid());
    }

    // Check that we can copy back to the device
    MockStateData<Ownership::value, MemSpace::host> host_states;
    resize(&host_states, 16);
    ASSERT_NO_THROW(device_states = copy_to_device_test(host_states));
    EXPECT_EQ(16, device_states.size());

    host_states = {};
    resize(&host_states, 8);
    auto host_state_ref = make_ref(host_states);
    ASSERT_NO_THROW(device_states = copy_to_device_test(host_state_ref));
    EXPECT_EQ(8, device_states.size());

    host_states = {};
    resize(&host_states, 4);
    auto host_state_cref = make_ref(host_states);
    ASSERT_NO_THROW(device_states = copy_to_device_test(host_state_cref));
    EXPECT_EQ(4, device_states.size());

    MockStateData<Ownership::reference, MemSpace::device> device_state_ref;
    device_state_ref = reference_device_test(device_states);
    EXPECT_EQ(4, device_state_ref.size());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

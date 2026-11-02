//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ObserverPtr.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/ObserverPtr.hh"

#include <algorithm>
#include <iterator>
#include <type_traits>

#include "corecel/data/DeviceVector.hh"

#include "ObserverPtr.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

struct ExampleStruct
{
    int foo{0};
    int bar{0};
};

struct ExampleStructDerived : ExampleStruct
{
    int baz{0};
};

class ObserverPtrTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(ObserverPtrTest, host)
{
    using VecInt = std::vector<int>;
    VecInt some_data = {1, 2, 3, 4};

    auto obs = make_observer(some_data.data());
    EXPECT_TRUE(
        (std::is_same_v<decltype(obs), ObserverPtr<int, MemSpace::host>>));
    EXPECT_TRUE(obs);
    EXPECT_TRUE(obs == obs);
    EXPECT_TRUE(obs != nullptr);
    EXPECT_EQ(some_data.data() + 2, &obs[2]);
    EXPECT_EQ(1, *obs);
    EXPECT_EQ(1, *obs.get());

    // Release clears the pointer
    EXPECT_EQ(1, *obs.release());
    EXPECT_EQ(nullptr, obs.get());
    EXPECT_TRUE(obs == nullptr);

    auto cobs = make_observer(const_cast<int const*>(some_data.data()));
    EXPECT_TRUE((
        std::is_same_v<decltype(cobs), ObserverPtr<int const, MemSpace::host>>));
    EXPECT_EQ(1, *cobs);

#ifdef CELERITAS_SHOULD_NOT_COMPILE
    // Const correctness: should not convert from const to mutable
    ObserverPtr<int>{cobs};
#endif
}

TEST_F(ObserverPtrTest, TEST_IF_CELER_DEVICE(device))
{
    int const test_data[] = {1, 1, 2, 3, 5, 8};
    DeviceVector<int> inp_dv(std::size(test_data));
    DeviceVector<int> out_dv(std::size(test_data));
    std::vector<int> out(std::size(test_data), -1);

    auto src = make_observer(inp_dv);
    auto dst = make_observer(out_dv);

#ifdef CELERITAS_SHOULD_NOT_COMPILE
    // Assigning a device pointer on host is prohibited
    *src = 2;
    src[2] = 3;
#endif
    // Pointer cast can be used to access without memspace checking
    EXPECT_NE(nullptr, static_cast<int*>(src));

    // Test device pointer with manual kernel
    inp_dv.copy_to_device(make_span(test_data));
    copy_test(src, dst, inp_dv.size());
    out_dv.copy_to_host(make_span(out));
    EXPECT_VEC_EQ(make_span(test_data), out);

    // (clear data)
    std::fill(out.begin(), out.end(), -1);
    out_dv.copy_to_device(make_span(out));

    // Test device pointer with thrust kernel
    copy_thrust_test(src, dst, inp_dv.size());
    out_dv.copy_to_host(make_span(out));
    EXPECT_VEC_EQ(make_span(test_data), out);
}

TEST_F(ObserverPtrTest, host_member)
{
    ExampleStruct value{7, 9};
    auto struct_ptr = make_observer(&value);
    auto foo = make_observer(struct_ptr, &ExampleStruct::foo);
    EXPECT_TRUE(
        (std::is_same_v<decltype(foo), ObserverPtr<int, MemSpace::host>>));
    EXPECT_EQ(&value.foo, foo.get());
    EXPECT_EQ(7, *foo);
    EXPECT_TRUE(make_observer(struct_ptr, &ExampleStruct::foo)
                < make_observer(struct_ptr, &ExampleStruct::bar));

    ObserverPtr<ExampleStruct const> cstruct_ptr{struct_ptr};
    auto cfoo = make_observer(cstruct_ptr, &ExampleStruct::foo);
    EXPECT_TRUE((
        std::is_same_v<decltype(cfoo), ObserverPtr<int const, MemSpace::host>>));
}

TEST_F(ObserverPtrTest, host_derived_member)
{
    ExampleStructDerived esd;
    esd.foo = 3;
    esd.bar = 5;
    esd.baz = 11;

    auto esd_ptr = make_observer(&esd);
    {
        auto baz_ptr = make_observer(esd_ptr, &ExampleStructDerived::baz);
        EXPECT_EQ(11, *baz_ptr);
    }
    {
        auto foo_ptr = make_observer(esd_ptr, &ExampleStruct::foo);
        EXPECT_EQ(3, *foo_ptr);
    }
    {
        auto base_ptr = ObserverPtr<ExampleStruct>(esd_ptr);
#ifdef CELERITAS_SHOULD_NOT_COMPILE
        make_observer(base_ptr, &ExampleStructDerived::baz);
#endif
        // Since ESD::bar is ES::bar, this actually works:
        auto bar_ptr = make_observer(base_ptr, &ExampleStructDerived::bar);
        EXPECT_EQ(5, *bar_ptr);
    }
    {
        // Slice base class from derived
        auto es_cptr = ObserverPtr<ExampleStruct const>(esd_ptr);
        auto bar_ptr = make_observer(es_cptr, &ExampleStruct::bar);
        EXPECT_EQ(5, *bar_ptr);
    }
}

TEST_F(ObserverPtrTest, TEST_IF_CELER_DEVICE(device_member))
{
    DeviceVector<ExampleStruct> example_dv(3);
    auto example_ptr = make_observer(example_dv);
    auto foo = make_observer(example_ptr, &ExampleStruct::foo);
    static_assert(
        std::is_same_v<decltype(foo), ObserverPtr<int, MemSpace::device>>);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

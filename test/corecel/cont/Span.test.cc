//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Span.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Span.hh"

#include <iomanip>
#include <sstream>
#include <type_traits>

#include "corecel/cont/Array.hh"
#include "corecel/io/StreamUtils.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(SpanTest, fixed_size_zero)
{
    Span<int, 0> empty_span;
    EXPECT_EQ(nullptr, empty_span.data());
    EXPECT_EQ(0, empty_span.size());
    EXPECT_TRUE(empty_span.empty());
    EXPECT_EQ(sizeof(int*), sizeof(empty_span));
    EXPECT_EQ("{}", stream_to_string(empty_span));

    {
        auto templ_subspan = empty_span.subspan<0, 0>();
        EXPECT_TRUE((
            std::is_same<decltype(empty_span), decltype(templ_subspan)>::value));
    }
    {
        auto templ_subspan = empty_span.first<0>();
        EXPECT_TRUE((
            std::is_same<decltype(empty_span), decltype(templ_subspan)>::value));
    }
    {
        auto templ_subspan = empty_span.last<0>();
        EXPECT_TRUE((
            std::is_same<decltype(empty_span), decltype(templ_subspan)>::value));
    }

    // Test copy constructor
    Span<int, 0> other_span{empty_span};
    EXPECT_EQ(nullptr, other_span.data());
    EXPECT_EQ(0, other_span.size());
    empty_span = other_span;

    // Test dynamic conversion
    Span<int> dynamic{empty_span};
    EXPECT_EQ(nullptr, dynamic.data());
    EXPECT_EQ(0, dynamic.size());

    // Test type conversion
    Span<int const, 0> const_span{empty_span};
    EXPECT_EQ(0, const_span.size());

    // Test type conversion
    Span<int const> const_dynamic{empty_span};
    EXPECT_EQ(0, const_dynamic.size());
    Span<int const, 0> also_fixed{const_dynamic};
    EXPECT_EQ(0, also_fixed.size());

    // Removing const must not be allowed
    EXPECT_FALSE((std::is_constructible_v<Span<int, 0>, Span<int const, 0>>))
        << "const->mutable span is prohibited";
    EXPECT_FALSE((std::is_constructible_v<Span<int>, Span<int const>>))
        << "const->mutable dynamic span is prohibited";
    // Incompatible fixed extents must not be allowed
    EXPECT_FALSE((std::is_constructible_v<Span<int, 2>, Span<int, 3>>))
        << "fixed span with different extent is prohibited";

    // Test empty iterator construction
    int const* null_int{nullptr};
    Span<int const, 0> null_span(null_int, null_int);
    EXPECT_EQ(0, null_span.size());
}

TEST(SpanTest, fixed_size)
{
    int local_data[] = {123, 456};
    Span<int, 2> local_span(local_data);
    EXPECT_EQ(sizeof(int*), sizeof(local_span));
    EXPECT_EQ("{123,456}", stream_to_string(local_span));

    EXPECT_EQ(local_data, local_span.begin());
    EXPECT_EQ(local_data + 2, local_span.end());
    EXPECT_EQ(local_data, local_span.data());
    EXPECT_EQ(2, local_span.size());
    EXPECT_EQ(2 * sizeof(int), local_span.size_bytes());
    EXPECT_FALSE(local_span.empty());
    EXPECT_EQ(&local_data[1], &(local_span[1]));
    EXPECT_EQ(123, local_span.front());
    EXPECT_EQ(456, local_span.back());

    auto templ_subspan = local_span.subspan<1>();
    EXPECT_TRUE((std::is_same<Span<int, 1>, decltype(templ_subspan)>::value));
    EXPECT_EQ(local_data + 1, templ_subspan.data());

    auto func_subspan = local_span.subspan(1, 1);
    EXPECT_EQ(1, func_subspan.size());
    EXPECT_EQ(local_data + 1, func_subspan.data());

    // Test copy constructor
    Span<int, 2> other_span{local_span};
    EXPECT_EQ(local_data, other_span.data());
    EXPECT_EQ(2, other_span.size());
    local_span = other_span;

    // Test dynamic conversion
    Span<int> dynamic{local_span};
    EXPECT_EQ(local_data, dynamic.data());
    EXPECT_EQ(2, dynamic.size());

    // Test type conversion
    Span<int const, 2> const_span{local_span};
    EXPECT_EQ(local_data, const_span.data());
    EXPECT_EQ(2, const_span.size());

    // Test type conversion
    Span<int const> const_dynamic{local_span};
    EXPECT_EQ(local_data, const_dynamic.data());
    EXPECT_EQ(2, const_dynamic.size());

    // Test pointer constructor
    Span<int, 2> ptr_span(local_data, local_data + 2);
    EXPECT_EQ(local_data, ptr_span.data());
}

TEST(SpanTest, implicit_const_conversion)
{
    // Test that Span<int, 3> implicitly converts to Span<int const>
    // (i.e., the converting constructor is not explicit)
    int local_data[] = {1, 2, 3};
    Span<int, 3> span(local_data);

    Span<int const> const_span = span;
    EXPECT_EQ(local_data, const_span.data());
    EXPECT_EQ(3, const_span.size());

    // Should be able to convert back *explicitly* to fixed-size
    Span<int const, 3> const_fixed_span{const_span};
    EXPECT_EQ(local_data, const_fixed_span.data());
    EXPECT_EQ(3, const_fixed_span.size());

    // Converting dynamic to wrong size should result in assertion failure
    if constexpr (CELERITAS_DEBUG)
    {
        EXPECT_THROW((Span<int const, 2>{const_span}), DebugError);
        EXPECT_THROW((Span<int const, 4>{const_span}), DebugError);
    }
}

TEST(SpanTest, dynamic_size)
{
    int local_data[] = {123, 456, 789};
    Span<int> local_span(local_data);
    EXPECT_EQ(sizeof(int*) + sizeof(std::size_t), sizeof(local_span));
    EXPECT_EQ("{123,456,789}", stream_to_string(local_span));

    EXPECT_EQ(local_data, local_span.begin());
    EXPECT_EQ(local_data + 3, local_span.end());
    EXPECT_EQ(local_data, local_span.data());
    EXPECT_EQ(3, local_span.size());
    EXPECT_EQ(3 * sizeof(int), local_span.size_bytes());
    EXPECT_FALSE(local_span.empty());
    EXPECT_EQ(&local_data[1], &(local_span[1]));
    EXPECT_EQ(123, local_span.front());
    EXPECT_EQ(789, local_span.back());

    {
        auto templ_subspan = local_span.subspan<1>();
        EXPECT_TRUE((std::is_same<Span<int>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data + 1, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.subspan<1, 2>();
        EXPECT_TRUE(
            (std::is_same<Span<int, 2>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data + 1, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.first<2>();
        EXPECT_TRUE(
            (std::is_same<Span<int, 2>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.first(2);
        EXPECT_TRUE((std::is_same<Span<int>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.last<2>();
        EXPECT_TRUE(
            (std::is_same<Span<int, 2>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data + 1, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.last(2);
        EXPECT_TRUE((std::is_same<Span<int>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data + 1, templ_subspan.data());
        EXPECT_EQ(2, templ_subspan.size());
    }
    {
        auto templ_subspan = local_span.last(3);
        EXPECT_TRUE((std::is_same<Span<int>, decltype(templ_subspan)>::value));
        EXPECT_EQ(local_data, templ_subspan.data());
        EXPECT_EQ(3, templ_subspan.size());
    }
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(local_span.last(4), DebugError);
        EXPECT_THROW(local_span.last<4>(), DebugError);
    }

    auto func_subspan = local_span.subspan(1);
    EXPECT_EQ(local_data + 1, func_subspan.data());
    EXPECT_EQ(2, func_subspan.size());

    // Test copy constructor
    Span<int> other_span{local_span};
    EXPECT_EQ(local_data, other_span.data());
    EXPECT_EQ(3, other_span.size());
    local_span = other_span;

    // Test dynamic conversion
    Span<int> dynamic{local_span};
    EXPECT_EQ(local_data, dynamic.data());
    EXPECT_EQ(3, dynamic.size());

    // Test type conversion
    Span<int const> const_dynamic{local_span};
    EXPECT_EQ(local_data, const_dynamic.data());
    EXPECT_EQ(3, const_dynamic.size());

    // Test type and size conversion
    Span<int const, 3> const_span{local_span};
    EXPECT_EQ(local_data, const_span.data());
    EXPECT_EQ(3, const_span.size());

    // Test pointer constructor
    Span<int> ptr_span(local_data, local_data + 3);
    EXPECT_EQ(local_data, ptr_span.data());
}

TEST(SpanTest, io_manip)
{
    int local_data[] = {123, 456, 789};
    Span<int> local_span(local_data);

    {
        std::ostringstream os;
        os << std::setw(20) << local_span;
        EXPECT_EQ("{   123,  456,  789}", os.str());
    }
    {
        std::ostringstream os;
        os << std::setw(3) << local_span;
        EXPECT_EQ("{123,456,789}", os.str());
    }
    {
        std::ostringstream os;
        os << std::setw(20) << std::left << local_span;
        EXPECT_EQ("{123   ,456  ,789  }", os.str());
    }
}

TEST(SpanTest, make_span)
{
    {
        using VecInt = std::vector<int>;
        VecInt values = {1, 2, 3};
        auto span = make_span(values);
        EXPECT_TRUE((std::is_same_v<Span<int>, decltype(span)>));
        EXPECT_VEC_EQ(values, span);

        auto cspan = make_span(const_cast<VecInt const&>(values));
        EXPECT_TRUE((std::is_same_v<Span<int const>, decltype(cspan)>));
    }
    {
        using ArrInt3 = Array<int, 3>;
        ArrInt3 values = {1, 2, 3};
        auto span = make_span(values);
        EXPECT_TRUE((std::is_same_v<Span<int, 3>, decltype(span)>))
            << demangled_type(span);
        EXPECT_VEC_EQ(values, span);

        auto cspan = make_span(const_cast<ArrInt3 const&>(values));
        EXPECT_TRUE((std::is_same_v<Span<int const, 3>, decltype(cspan)>))
            << demangled_type(span);
    }
    {
        using ArrInt3 = Array<int, 3>;
        using VecArrInt3 = std::vector<ArrInt3>;
        VecArrInt3 values = {{1, 2, 3}, {4, 5, 6}};
        auto span = make_span(values);
        EXPECT_TRUE((std::is_same_v<Span<ArrInt3>, decltype(span)>))
            << demangled_type(span);
        EXPECT_VEC_EQ(values, span);

        auto cspan = make_span(const_cast<VecArrInt3 const&>(values));
        EXPECT_TRUE((std::is_same_v<Span<ArrInt3 const>, decltype(cspan)>))
            << demangled_type(span);
    }
}

TEST(SpanTest, array_conversion)
{
    using ArrInt3 = Array<int, 3>;
    ArrInt3 arr = {1, 2, 3};
    ArrInt3 const carr = {4, 5, 6};

    // Implicit copy-initialization for both fixed and dynamic extents
    Span<int, 3> fixed = arr;
    EXPECT_EQ(arr.data(), fixed.data());
    EXPECT_EQ(3, fixed.size());

    Span<int const, 3> cfixed = arr;
    EXPECT_EQ(arr.data(), cfixed.data());

    Span<int const, 3> cfixed2 = carr;
    EXPECT_EQ(carr.data(), cfixed2.data());

    Span<int> dyn = arr;
    EXPECT_EQ(arr.data(), dyn.data());
    EXPECT_EQ(3, dyn.size());

    Span<int const> cdyn = carr;
    EXPECT_EQ(carr.data(), cdyn.data());
    EXPECT_EQ(3, cdyn.size());

    // All size-compatible conversions are implicit in all standards
    EXPECT_TRUE((std::is_convertible_v<ArrInt3&, Span<int, 3>>));
    EXPECT_TRUE((std::is_convertible_v<ArrInt3&, Span<int>>));
    EXPECT_TRUE((std::is_convertible_v<ArrInt3 const&, Span<int const, 3>>));
    EXPECT_TRUE((std::is_convertible_v<ArrInt3 const&, Span<int const>>));

    // Wrong fixed extent: not constructible
    EXPECT_FALSE((std::is_constructible_v<Span<int, 2>, ArrInt3&>))
        << "Array to fixed Span with different extent is prohibited";

    // Removing const must not be allowed
    EXPECT_FALSE((std::is_constructible_v<Span<int>, ArrInt3 const&>))
        << "const Array to mutable Span is prohibited";
    EXPECT_FALSE((std::is_constructible_v<Span<int, 3>, ArrInt3 const&>))
        << "const Array to mutable fixed Span is prohibited";
}

TEST(SpanTest, array_deduction)
{
    Array<int, 3> arr = {1, 2, 3};
    Array<int, 3> const carr = {4, 5, 6};

    // Mutable Array -> Span<T, N>
    Span mspan{arr};
    EXPECT_TRUE((std::is_same_v<Span<int, 3>, decltype(mspan)>));
    EXPECT_EQ(arr.data(), mspan.data());
    EXPECT_EQ(3, mspan.size());

    // Const Array -> Span<T const, N>
    Span cspan{carr};
    EXPECT_TRUE((std::is_same_v<Span<int const, 3>, decltype(cspan)>));
    EXPECT_EQ(carr.data(), cspan.data());
    EXPECT_EQ(3, cspan.size());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

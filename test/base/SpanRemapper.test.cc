//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SpanRemapper.test.cc
//---------------------------------------------------------------------------//
#include "base/SpanRemapper.hh"

#include <type_traits>
#include "gtest/Main.hh"
#include "gtest/Test.hh"

using celeritas::make_span;
using celeritas::make_span_remapper;
using celeritas::span;
using celeritas::SpanRemapper;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(SpanRemapperTest, host_host)
{
    std::vector<int> src_data(20);
    std::vector<int> dst_data(src_data.size());

    auto remap_span
        = make_span_remapper(make_span(src_data), make_span(dst_data));

#if CELERITAS_DEBUG
    {
        // Can't remap data outside of the 'src' range
        std::vector<int> other_data(20);
        EXPECT_THROW(remap_span(make_span(other_data)), celeritas::DebugError);
    }
#endif

    {
        // Remap entire span
        auto dst = remap_span(make_span(src_data));
        EXPECT_EQ(dst_data.data(), dst.data());
        EXPECT_EQ(dst_data.size(), dst.size());
    }

    {
        // Remap empty span
        auto dst = remap_span(span<int>{});
        EXPECT_EQ(0, dst.size());
    }

    {
        // Remap subspan
        span<int> src(&src_data[3], &src_data[10]);
        auto      dst = remap_span(src);
        EXPECT_EQ(3, dst.data() - dst_data.data());
        EXPECT_EQ(10 - 3, dst.size());
    }
}

TEST(SpanRemapperTest, host_host_const)
{
    std::vector<int> src_data(20);
    std::vector<int> dst_data(src_data.size());

    // (int, int)(const int) -> const int
    {
        auto remap_span
            = make_span_remapper(make_span(src_data), make_span(dst_data));

        span<const int> src(&src_data[3], &src_data[10]);
        auto            dst = remap_span(src);
        EXPECT_EQ(3, dst.data() - dst_data.data());
        EXPECT_EQ(10 - 3, dst.size());
        EXPECT_TRUE((std::is_same<decltype(dst), span<int>>::value));
    }

    // (int, const int)(int) -> const int
    {
        auto remap_span = make_span_remapper(
            make_span(src_data),
            span<const int>(dst_data.data(), dst_data.size()));

        span<int> src(&src_data[3], &src_data[10]);
        auto      dst = remap_span(src);
        EXPECT_TRUE((std::is_same<decltype(dst), span<const int>>::value));
    }
}

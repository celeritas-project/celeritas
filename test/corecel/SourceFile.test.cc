//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/SourceFile.test.cc
//---------------------------------------------------------------------------//
#include "corecel/SourceFile.hh"

#include <string_view>

#include "corecel/io/StringUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
using detail::source_dir_offset;

// POSIX paths
static_assert(source_dir_offset("/w/celeritas/src/corecel/Assert.hh") == 17);
static_assert(source_dir_offset("/w/celeritas/test/corecel/A.test.cc") == 18);
static_assert(source_dir_offset("/w/celeritas/app/celer-sim/x.cc") == 17);
static_assert(source_dir_offset("src/corecel/Assert.hh") == 4);
static_assert(source_dir_offset("./src/corecel/Assert.hh") == 6);
// Last matching component wins
static_assert(source_dir_offset("/src/celeritas/test/corecel/A.cc") == 20);
// Windows paths
static_assert(source_dir_offset("C:\\a\\celeritas\\src\\corecel\\A.hh") == 19);
// Not a path component: no trimming
static_assert(source_dir_offset("/w/mysrc/a.hh") == 0);
static_assert(source_dir_offset("/w/srcs/a.hh") == 0);
static_assert(source_dir_offset("/w/testing/a.hh") == 0);
static_assert(source_dir_offset("/other/project/foo.cc") == 0);
static_assert(source_dir_offset("foo.cc") == 0);
static_assert(source_dir_offset("") == 0);

// Full trimming on fixed paths, as done by the macro for __FILE__
struct PosixPath
{
    static constexpr char const* str()
    {
        return "/w/celeritas/src/corecel/io/Logger.hh";
    }
};
struct WindowsPath
{
    static constexpr char const* str()
    {
        return "D:\\a\\celeritas\\celeritas\\src\\corecel\\io\\Logger.hh";
    }
};
struct ExternalPath
{
    static constexpr char const* str() { return "/other/project/foo.cc"; }
};

constexpr char const* in_constexpr_function()
{
    return CELER_SOURCE_FILE;
}

// The macro is itself a constant expression
static_assert(in_constexpr_function()[0] == 'c');

template<class T>
char const* in_template(T)
{
    return CELER_SOURCE_FILE;
}

struct InMemberInitializer
{
    char const* file = CELER_SOURCE_FILE;
};

//---------------------------------------------------------------------------//
}  // namespace

TEST(SourceFileTest, fixed_paths)
{
    using detail::source_file;
    EXPECT_STREQ("corecel/io/Logger.hh", source_file<PosixPath>());
    EXPECT_STREQ("corecel/io/Logger.hh", source_file<WindowsPath>());
    EXPECT_STREQ("/other/project/foo.cc", source_file<ExternalPath>());
    // Identical results share storage
    EXPECT_EQ(source_file<PosixPath>(), source_file<WindowsPath>());
}

TEST(SourceFileTest, trimmed)
{
    std::string_view expected{"corecel/SourceFile.test.cc"};
    EXPECT_EQ(expected, CELER_SOURCE_FILE);
    EXPECT_EQ(expected, in_constexpr_function());
    EXPECT_EQ(expected, in_template(1));
    EXPECT_EQ(expected, InMemberInitializer{}.file);
}

TEST(SourceFileTest, stable_pointer)
{
    // The same file yields a single static array, so pointers compare equal
    char const* a = CELER_SOURCE_FILE;
    char const* b = in_template(1.0);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a, InMemberInitializer{}.file);
}

TEST(SourceFileTest, no_full_path)
{
    // The full path (if any) must not be the one reported
    std::string_view result{CELER_SOURCE_FILE};
    EXPECT_EQ(std::string_view::npos, result.find("/test/"));
    EXPECT_EQ(std::string_view::npos, result.find("\\test\\"));
    EXPECT_TRUE(starts_with(result, "corecel/"));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

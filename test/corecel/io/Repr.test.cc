//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Repr.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/Repr.hh"

#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/StringUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
struct FreeFunctionValue
{
    int value;
};

struct FreeFunctionRepr
{
    FreeFunctionValue const& obj;

    friend std::ostream& operator<<(std::ostream& os,
                                    FreeFunctionRepr const& repr)
    {
        return os << "external{" << repr.obj.value << '}';
    }
};

inline FreeFunctionRepr repr(FreeFunctionValue const& obj)
{
    return {obj};
}

namespace test
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

template<class T>
std::string repr_to_string(T const& obj)
{
    std::ostringstream os;
    os << repr(obj);
    return os.str();
}

struct HiddenFriendValue
{
    int value;

    struct Repr
    {
        HiddenFriendValue const& obj;

        friend std::ostream& operator<<(std::ostream& os, Repr const& repr)
        {
            return os << "hidden{" << repr.obj.value << '}';
        }
    };

    friend Repr repr(HiddenFriendValue const& obj) { return {obj}; }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(ReprTest, fundamental)
{
    EXPECT_EQ("1.25f", repr_to_string(1.25f));

    EXPECT_EQ("1.25", repr_to_string(1.25));
    EXPECT_EQ("1", repr_to_string(1));
    EXPECT_EQ("1ull", repr_to_string(1ull));
    EXPECT_EQ("1ll", repr_to_string(1ll));
    EXPECT_EQ("'a'", repr_to_string('a'));
    EXPECT_EQ("'\\x61'", repr_to_string(static_cast<unsigned char>('a')));
    EXPECT_EQ("'\\xff'", repr_to_string('\xff'));
}

TEST(ReprTest, string)
{
    EXPECT_EQ("\"abcd\\a\"", repr_to_string("abcd\a"));
    EXPECT_EQ("\"abcd\\a\"", repr_to_string(std::string("abcd\a")));

    auto longstr = repr_to_string(std::string(100, 'x'));
    EXPECT_TRUE(starts_with(longstr, "R\"(xxx")) << longstr;
    EXPECT_TRUE(ends_with(longstr, "xxx)\"")) << longstr;
}

TEST(ReprTest, overloads)
{
    EXPECT_EQ("hidden{42}", repr_to_string(HiddenFriendValue{42}));
    EXPECT_EQ("external{17}", repr_to_string(FreeFunctionValue{17}));
}

TEST(ReprTest, container)
{
    EXPECT_EQ("{1, 2, 3, 4}", repr_to_string(std::vector<int>{1, 2, 3, 4}));
    EXPECT_EQ("{100l, 200l}", repr_to_string(Array<long, 2>{100, 200}));
    EXPECT_EQ("{0, 1, 2, 3, 4, 5, 6, 7, 8,}",
              repr_to_string(Array<int, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ("{-1, 0, 1}", repr_to_string(std::set<int>{-1, 0, 1}));

    unsigned int uints[] = {11, 22};
    EXPECT_EQ("{11u, 22u}", repr_to_string(make_span(uints)));

    char const* const cstrings[] = {"one", "three", "five"};
    EXPECT_EQ("{\"one\", \"three\", \"five\"}", repr_to_string(cstrings));

    std::string const strings[] = {"a", "", "special\nchars\t"};
    EXPECT_EQ("{\"a\", \"\", \"special\\nchars\\t\"}", repr_to_string(strings));

    std::vector<std::string> long_vec(20, std::string(10, 'x'));
    auto long_vec_str = repr_to_string(long_vec);
    EXPECT_TRUE(ends_with(long_vec_str, "\",}")) << long_vec_str;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

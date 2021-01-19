//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Join.test.cc
//---------------------------------------------------------------------------//
#include "base/Join.hh"

#include <fstream>
#include "celeritas_test.hh"
#include "base/Range.hh"

using celeritas::join;
using celeritas::join_stream;

//---------------------------------------------------------------------------//
// Helper classes
//---------------------------------------------------------------------------//

struct Moveable
{
    std::string msg;
    int*        counter;

    Moveable(std::string m, int* c) : msg(std::move(m)), counter(c)
    {
        CELER_EXPECT(counter);
    }

    Moveable(Moveable&& rhs) : msg(std::move(rhs.msg)), counter(rhs.counter)
    {
        ++(*counter);
    }

    Moveable& operator=(Moveable&& rhs)
    {
        msg     = std::move(rhs.msg);
        counter = rhs.counter;
        ++(*counter);
        return *this;
    }

    // Delete copy and copy assign
    Moveable(const Moveable& rhs) = delete;
    Moveable& operator=(const Moveable& rhs) = delete;
};

std::ostream& operator<<(std::ostream& os, const Moveable& m)
{
    return os << m.msg;
}

struct transform_functor
{
    int* counter = nullptr;

    template<class T>
    T operator()(const T& rhs)
    {
        if (counter)
            ++(*counter);
        return rhs + 1;
    }
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class JoinTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Typical use case
TEST_F(JoinTest, typical)
{
    std::vector<int> vals = {3, 4, 5};
    EXPECT_EQ("3, 4, 5", to_string(join(vals.begin(), vals.end(), ", ")));

    vals = {3};
    EXPECT_EQ("3", to_string(join(vals.begin(), vals.end(), ":")));

    std::ostringstream os;
    vals = {101, 202, 303};
    os << join(vals.begin(), vals.end(), ", ") << "!";
    EXPECT_EQ("101, 202, 303!", os.str());
}

//---------------------------------------------------------------------------//
// Demonstrates that Join doesn't have to allocate the joined string
TEST_F(JoinTest, DISABLED_ginormous)
{
    std::ofstream out(this->make_unique_filename(".txt"));

    auto r = celeritas::range<std::size_t>(1e7);
    out << join(r.begin(), r.end(), "\n");
}

//---------------------------------------------------------------------------//
TEST_F(JoinTest, transformed)
{
    std::vector<int> vals = {3, 4, 5};
    EXPECT_EQ("6, 8, 10",
              to_string(join(vals.begin(), vals.end(), ", ", [](int v) {
                  return v * 2;
              })));

    int transform_ctr = 0;
    EXPECT_EQ(
        "4,5,6",
        to_string(join(
            vals.begin(), vals.end(), ",", transform_functor{&transform_ctr})));
    EXPECT_EQ(3, transform_ctr);
}

//---------------------------------------------------------------------------//
TEST_F(JoinTest, streamed)
{
    // Join using stream operator
    using Pair_t              = std::pair<int, double>;
    std::vector<Pair_t> pairs = {{3, 1.5}, {4, -1.0}, {5, 1e9}};

    // >>> MOVE CONJUNCTION

    int      counter = 0;
    Moveable conjunction{", ", &counter};

    // NOTE: if the Join implementation class uses a const std::string& for the
    // conjunction, the code below may crash because ", " will be passed as a
    // temporary that is destroyed before `j` uses it. But now Join uses
    // perfect forwarding to ensure that the conjunction is captured.
    auto j = join_stream(pairs.begin(),
                         pairs.end(),
                         std::move(conjunction),
                         [](std::ostream& os, const Pair_t& p) {
                             os << p.first << "->" << p.second;
                         });
    // Change the conjunction; shouldn't matter to the first output
    int unused_counter = 0;
    conjunction        = Moveable{"@", &unused_counter};

    EXPECT_EQ("3->1.5, 4->-1, 5->1e+09", to_string(j));
    EXPECT_EQ(1, counter);

    // >>> REFERENCE CONJUNCTION

    auto j2 = join(pairs.begin(), pairs.end(), conjunction, [](const Pair_t& p) {
        return p.first * p.second;
    });
    // Change the passed value
    conjunction = Moveable{"|", &counter};

    EXPECT_EQ("4.5|-4|5e+09", to_string(j2));
    EXPECT_EQ(2, counter);
}

//---------------------------------------------------------------------------//

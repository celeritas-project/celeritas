//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TypeDemangler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/TypeDemangler.hh"

#include "celeritas_test.hh"

namespace tdtest
{
//---------------------------------------------------------------------------//

template<class T>
struct FlorbyDorb
{
};

struct Zanzibar
{
};

struct JapaneseIsland
{
    virtual ~JapaneseIsland() {}
};

struct Honshu : public JapaneseIsland
{
};

struct Hokkaido : public JapaneseIsland
{
};

void do_stuff() {}

template<class T>
void do_templated_stuff(T)
{
}

//---------------------------------------------------------------------------//
}  // namespace tdtest

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

template<class F>
std::string get_templated_funcname(F)
{
    return TypeDemangler<F>()();
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(TypeDemanglerTest, demangled_typeid_name)
{
    std::string int_type = demangled_typeid_name(typeid(int).name());
    std::string flt_type = demangled_typeid_name(typeid(float).name());

    EXPECT_NE(int_type, flt_type);

#ifdef __GNUG__
    EXPECT_EQ("int", int_type);
    EXPECT_EQ("float", flt_type);
#endif

    EXPECT_EQ(int_type, demangled_type(3));
    int const myint{1234};
    EXPECT_EQ(int_type, demangled_type(myint));
}

TEST(TypeDemanglerTest, static_types)
{
    using namespace tdtest;

    TypeDemangler<FlorbyDorb<Zanzibar>> demangle_type;

    std::string fdz_type = demangle_type();
    EXPECT_NE(fdz_type, TypeDemangler<FlorbyDorb<Hokkaido>>()());

#ifdef __GNUG__
    EXPECT_EQ("tdtest::FlorbyDorb<tdtest::Zanzibar>", fdz_type);
#endif

    get_templated_funcname(do_stuff);
    get_templated_funcname(&do_templated_stuff<int>);
}

TEST(TypeDemanglerTest, dynamic)
{
    using namespace tdtest;

    TypeDemangler<JapaneseIsland> demangle;
    Honshu const honshu{};
    Hokkaido const hokkaido{};
    JapaneseIsland const& hon_ptr = honshu;
    JapaneseIsland const& hok_ptr = hokkaido;

    EXPECT_EQ(demangle(honshu), demangle(hon_ptr));
    EXPECT_EQ(demangle(hokkaido), demangle(hok_ptr));
    EXPECT_NE(demangle(JapaneseIsland()), demangle(hon_ptr));

#ifdef __GNUG__
    EXPECT_EQ("tdtest::Hokkaido", demangle(hok_ptr));
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

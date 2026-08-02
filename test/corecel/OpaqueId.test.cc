//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/OpaqueId.test.cc
//---------------------------------------------------------------------------//
#include "corecel/OpaqueId.hh"

#include <cstdint>
#include <type_traits>
#include <unordered_map>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

struct TestInstantiator;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
template<class T>
class OpaqueIdTypedTest : public ::testing::Test
{
};

using IntTypes = ::testing::Types<unsigned char, unsigned int, std::size_t>;
TYPED_TEST_SUITE(OpaqueIdTypedTest, IntTypes, );

TYPED_TEST(OpaqueIdTypedTest, unassigned)
{
    using Int_t = TypeParam;
    using Id_t = OpaqueId<TestInstantiator, Int_t>;
    constexpr auto sizemax = static_cast<Int_t>(-1);

    Id_t unassigned;
    EXPECT_FALSE(unassigned);
    EXPECT_TRUE(!unassigned);
    EXPECT_EQ(unassigned, unassigned);
    EXPECT_EQ(unassigned, Id_t{});
    EXPECT_TRUE(unassigned == nullid);
    EXPECT_TRUE(nullid == unassigned);
    EXPECT_FALSE(unassigned != nullid);
    EXPECT_FALSE(nullid != unassigned);
    EXPECT_EQ(sizemax, Id_t{}.unchecked_get());
    EXPECT_EQ(std::hash<TypeParam>()(sizemax), std::hash<Id_t>()(unassigned));
    if constexpr (CELERITAS_DEBUG)
    {
        EXPECT_THROW(unassigned.get(), DebugError);
        EXPECT_THROW(unassigned.value(), DebugError);
    }

    EXPECT_EQ("{}", stream_to_string(Id_t{}));
}

TYPED_TEST(OpaqueIdTypedTest, assigned)
{
    using Int_t = TypeParam;
    using Id_t = OpaqueId<TestInstantiator, Int_t>;

    Id_t assigned{123};
    EXPECT_TRUE(assigned);
    EXPECT_FALSE(!assigned);
    EXPECT_EQ(123, assigned.get());
    EXPECT_EQ(Int_t{123}, assigned.value());
    EXPECT_EQ(Int_t{123}, *assigned);
    EXPECT_NE(Id_t{}, assigned);
    EXPECT_EQ(assigned, assigned);
    EXPECT_FALSE(assigned == nullid);
    EXPECT_FALSE(nullid == assigned);
    EXPECT_TRUE(assigned != nullid);
    EXPECT_TRUE(nullid != assigned);
    EXPECT_EQ(std::hash<TypeParam>()(123), std::hash<Id_t>()(assigned));

    EXPECT_EQ("{123}", stream_to_string(assigned));

    assigned = nullid;
    EXPECT_FALSE(assigned);
}

TYPED_TEST(OpaqueIdTypedTest, traits)
{
    using Int_t = TypeParam;
    using Id_t = OpaqueId<TestInstantiator, Int_t>;

    EXPECT_TRUE(is_opaque_id_v<Id_t>);
    EXPECT_TRUE(is_opaque_id_v<Id_t const>);
    EXPECT_FALSE(is_opaque_id_v<Int_t>);

    EXPECT_TRUE((std::is_same_v<Int_t, MakeSize_t<Id_t>>));
    EXPECT_TRUE((std::is_same_v<Int_t, MakeSize_t<Id_t const>>));
    EXPECT_TRUE((std::is_trivially_copyable_v<Id_t>));
    EXPECT_TRUE((std::is_trivially_destructible_v<Id_t>));
    EXPECT_TRUE((std::is_default_constructible_v<std::hash<Id_t>>));
    EXPECT_TRUE((std::is_copy_constructible_v<std::hash<Id_t>>));
    EXPECT_TRUE((is_auto_ldg_v<Id_t>));
}

TYPED_TEST(OpaqueIdTypedTest, operators)
{
    using Id_t = OpaqueId<TestInstantiator, TypeParam>;
    constexpr auto nullid_value = detail::nullid_value<TypeParam>;
    constexpr auto largest_valid = nullid_value - 1;

    EXPECT_TRUE(Id_t{22} <= Id_t{23});
    EXPECT_TRUE(Id_t{22} < Id_t{23});
    EXPECT_TRUE(Id_t{23} > Id_t{22});
    EXPECT_TRUE(Id_t{23} >= Id_t{22});
    EXPECT_TRUE(Id_t{22} <= Id_t{23});

    EXPECT_FALSE(Id_t{23} <= Id_t{22});
    EXPECT_FALSE(Id_t{23} < Id_t{22});
    EXPECT_FALSE(Id_t{22} > Id_t{23});
    EXPECT_FALSE(Id_t{22} >= Id_t{23});
    EXPECT_FALSE(Id_t{23} <= Id_t{22});

    EXPECT_EQ(10, Id_t{22} - Id_t{12});
    EXPECT_EQ(-10, Id_t{12} - Id_t{22});
    EXPECT_EQ(Id_t{24}, Id_t{22} + 2);
    EXPECT_EQ(Id_t{24}, 2 + Id_t{22});
    EXPECT_EQ(Id_t{24}, Id_t{22} + std::size_t(2));
    EXPECT_EQ(Id_t{20}, Id_t{22} + std::ptrdiff_t(-2));
    EXPECT_EQ(Id_t{24}, Id_t{22} - (-2));
    EXPECT_EQ(Id_t{22}, Id_t{22} + 0);
    EXPECT_EQ(Id_t{22}, Id_t{22} - 0);
    EXPECT_EQ(Id_t{21}, Id_t{22} + (-1));
    EXPECT_EQ(Id_t{21}, (-1) + Id_t{22});
    EXPECT_EQ(Id_t{21}, Id_t{22} - 1);
    EXPECT_EQ(Id_t{0}, Id_t{1} - 1);
    EXPECT_EQ(Id_t{0}, Id_t{2} + (-2));
    EXPECT_EQ(Id_t{1}, ++Id_t{0});
    EXPECT_EQ(largest_valid - 1, *(--Id_t{largest_valid}));
    if constexpr (CELERITAS_DEBUG)
    {
        EXPECT_THROW(Id_t{} + (-2), DebugError);
        EXPECT_THROW(Id_t{} - 2, DebugError);
        EXPECT_THROW(Id_t{1} + (-2), DebugError);
        EXPECT_THROW((-2) + Id_t{1}, DebugError);
        EXPECT_THROW(Id_t{1} - 2, DebugError);
        EXPECT_THROW(--Id_t{}, DebugError);
        EXPECT_THROW(++Id_t{}, DebugError);
        EXPECT_THROW(++Id_t{largest_valid}, DebugError);
        // NOTE: since the operators take a signed integer, adding/subtracting
        // sizemax will cause it to overflow (UB)
    }

    Id_t id{0};
    Id_t old{id++};
    EXPECT_EQ(Id_t{1}, id);
    EXPECT_EQ(Id_t{0}, old);
}

TYPED_TEST(OpaqueIdTypedTest, ldg)
{
    using Id_t = OpaqueId<TestInstantiator, TypeParam>;

    Id_t const id{12};

    EXPECT_EQ(Id_t{12}, ldg(&id));
    EXPECT_TRUE((std::is_same_v<Id_t, decltype(ldg(&id))>));
}

TEST(OpaqueIdTest, multi_int)
{
    using UId8 = OpaqueId<TestInstantiator, std::uint_least8_t>;
    using UInt32 = std::uint_least32_t;
    using Int32 = std::make_signed_t<UInt32>;
    using Limits_t = std::numeric_limits<UInt32>;

    // Unassigned is always out-of-range
    EXPECT_FALSE(UId8{} < 0u);
    EXPECT_FALSE(UId8{} < UInt32(Limits_t::max()));
    EXPECT_FALSE(UId8{} < UInt32(255));
    EXPECT_FALSE(UId8{} < UInt32(256));
    EXPECT_FALSE(UId8{10} < UInt32(5));

    // Check 'true' conditions
    EXPECT_TRUE(UId8{254} < UInt32(Limits_t::max()));
    EXPECT_TRUE(UId8{254} < UInt32(255));
    EXPECT_TRUE(UId8{10} < UInt32(15));

    // Check less-or-equal
    EXPECT_FALSE(UId8{} <= UInt32(9));
    EXPECT_TRUE(UId8{253} <= UInt32(254));
    EXPECT_TRUE(UId8{254} <= UInt32(254));

    EXPECT_EQ(UId8{1}, UId8{254} - UInt32{253});
    EXPECT_EQ(UId8{1}, UId8{254} - Int32{253});
    EXPECT_EQ(UId8{1}, UId8{254} + Int32{-253});
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(UId8{3} + Int32{-4}, DebugError);
        // NOTE: UId8{3} + Int32{-256 - 2} is UB
    }
}

TEST(OpaqueIdTest, unordered_map)
{
    using IdT = OpaqueId<TestInstantiator>;

    std::unordered_map<IdT, int> ids;
    ids[IdT{2}] = 2;
    ids[IdT{4}] = 4;

    EXPECT_EQ(2, ids[IdT{2}]);
    EXPECT_EQ(4, ids[IdT{4}]);
}

TEST(OpaqueIdTest, id_cast)
{
    using IdT = OpaqueId<TestInstantiator, unsigned short int>;

#ifdef CELERITAS_SHOULD_NOT_COMPILE
    id_cast<IdT>("this shouldn't compile");
    id_cast<IdT>(1e4);  // nor should this
#endif

    EXPECT_EQ(3, id_cast<IdT>(3).unchecked_get());
    int const lvalue{254};
    EXPECT_EQ(lvalue, id_cast<IdT>(lvalue).unchecked_get());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(id_cast<IdT>(-12345678ll), DebugError);
        EXPECT_THROW(id_cast<IdT>(-1), DebugError);
        EXPECT_THROW(id_cast<IdT>(IdT{}.unchecked_get()), DebugError);
        EXPECT_THROW(
            id_cast<IdT>(static_cast<unsigned int>(IdT{}.unchecked_get()) + 1),
            DebugError);
        EXPECT_THROW(id_cast<IdT>(100000000l), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

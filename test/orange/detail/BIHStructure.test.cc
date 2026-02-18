//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHStructure.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHStructure.hh"

#include <utility>
#include <nlohmann/json.hpp>

#include "orange/detail/BIHBuilder.hh"
#include "orange/detail/BIHStructureIO.json.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
class BIHStructureTest : public ::celeritas::test::Test
{
  protected:
    BIHBuilder::SetLocalVolId implicit_vol_ids_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
};

//---------------------------------------------------------------------------//
TEST_F(BIHStructureTest, basic)
{
    BIHBuilder::VecBBox bboxes = {
        FastBBox::from_infinite(),
        {{0, 0, 0}, {1.6f, 1, 100}},
        {{1.2f, 0, 0}, {2.8f, 1, 100}},
        {{2.8f, 0, 0}, {5, 1, 100}},
        {{0, -1, 0}, {5, 0, 100}},
        {{0, -1, 0}, {5, 0, 100}},
    };

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);

    ref_storage_ = storage_;
    BIHStructure structure{bih_tree, ref_storage_};

    EXPECT_EQ(7, structure.tree().size());
    EXPECT_EQ(1, structure.inf_vol_ids().size());
    EXPECT_EQ(LocalVolumeId{0}, structure.inf_vol_ids().front());

    nlohmann::json j = structure;
    EXPECT_JSON_EQ(
        R"json({"inf_vol_ids":[0],"tree":[["i","x",[1,2],[2.799999952316284,0.0]],["i","x",[3,4],[1.600000023841858,1.2000000476837158]],["i","x",[5,6],[5.0,2.799999952316284]],["l",[1]],["l",[2]],["l",[4,5]],["l",[3]]]})json",
        j.dump(0));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas

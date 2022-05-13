//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/VolumeInserter.test.cc
//---------------------------------------------------------------------------//
#include "orange/construct/VolumeInserter.hh"

#include <fstream>

#include "celeritas_config.h"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "orange/construct/VolumeInput.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_JSON
#    include "orange/construct/VolumeInputIO.json.hh"
#endif

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class VolumeInserterTest : public celeritas_test::Test
{
  protected:
    VolumeData<Ownership::value, MemSpace::host>            volume_data_;
    SurfaceData<Ownership::value, MemSpace::host>           surface_data_;
    SurfaceData<Ownership::const_reference, MemSpace::host> surface_ref_;

    void SetUp() override
    {
        // Build enough fake surfaces for the five volumes problem
        auto types   = make_builder(&surface_data_.types);
        auto offsets = make_builder(&surface_data_.offsets);
        auto reals   = make_builder(&surface_data_.reals);

        for (auto st : {SurfaceType::sc,
                        SurfaceType::px,
                        SurfaceType::px,
                        SurfaceType::py,
                        SurfaceType::py,
                        SurfaceType::pz,
                        SurfaceType::pz,
                        SurfaceType::sc,
                        SurfaceType::px,
                        SurfaceType::px,
                        SurfaceType::py,
                        SurfaceType::sc})
        {
            types.push_back(st);
            offsets.push_back(OpaqueId<real_type>(surface_data_.reals.size()));
            reals.push_back(0);
        }

        surface_ref_ = make_const_ref(surface_data_);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(VolumeInserterTest, manual)
{
    VolumeInserter insert(surface_ref_, &volume_data_);

    {
        // Empty volume
        VolumeInput input;
        input.logic = {logic::ltrue};
        EXPECT_EQ(VolumeId{0}, insert(input));
        EXPECT_EQ(1, insert.max_logic_depth());
    }

    {
        // Volume with one face
        VolumeInput input;
        input.faces = {SurfaceId{0}};
        input.logic = {0};
        EXPECT_EQ(VolumeId{1}, insert(input));
        EXPECT_EQ(1, insert.max_logic_depth());
    }

    {
        // Volume with two joined face
        VolumeInput input;
        input.faces = {SurfaceId{0}, SurfaceId{1}};
        input.logic = {0, logic::lnot, 1, logic::land};
        EXPECT_EQ(VolumeId{2}, insert(input));
        EXPECT_EQ(2, insert.max_logic_depth());
    }

    {
        // Invalid definition (needs 'and'/'or')
        VolumeInput input;
        input.faces = {SurfaceId{0}, SurfaceId{1}};
        input.logic = {0, logic::lnot, 1};
        EXPECT_THROW(insert(input), RuntimeError);
    }
}

TEST_F(VolumeInserterTest, from_json)
{
    VolumeInserter insert(surface_ref_, &volume_data_);
    std::ifstream  infile(
        this->test_data_path("orange", "five-volumes.org.json"));

#if !CELERITAS_USE_JSON
    GTEST_SKIP() << "JSON is not enabled";
#else
    auto full_inp = nlohmann::json::parse(infile);

    VolumeId::size_type volid = 0;
    for (const auto& vol_inp : full_inp["universes"][0]["cells"])
    {
        auto id = insert(vol_inp.get<VolumeInput>());
        EXPECT_EQ(volid, id.unchecked_get());
        ++volid;
    }
#endif
}

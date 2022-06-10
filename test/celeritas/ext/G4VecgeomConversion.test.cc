//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/G4VecgeomConverter.hh"

//#include <algorithm>
//#include "corecel/Types.hh"
//#include "corecel/cont/Range.hh"
//#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/VecgeomData.hh"
#include "celeritas/geo/GeoParams.hh"

#include "Geant4/G4GDMLParser.hh"
#include "celeritas_test.hh"

using namespace celeritas;
template<MemSpace M>
using PlacedVolumeT = typename detail::VecgeomTraits<M>::PlacedVolume;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The \e four-levels.gdml file is used t   o construct a Geant4 model,
 * which is later converted into a corresponding Vecgeom model.
 */
class G4VecgeomConvTest : public celeritas_test::Test
{
  protected:
    void SetUp() override
    {
        // GeantSetup setup = {"four_levels.gdml", GeantSetupOptions{}};
        G4GDMLParser parser;
        // parser.SetOverlapCheck(true);
        parser.Read(filename_.c_str(), false);

        // TODO: use celeritas way to deal with placed volumes in host/device
        // host_world_ = vecgeom::GeoManager::Instance().GetWorld();
        geom_ = std::make_shared<GeoParams>(parser.GetWorldVolume());
    }

    std::string filename_
        = this->test_data_path("celeritas", "four-levels.gdml");

  public:
    void PrintContent() const;

    //  private:
    PlacedVolumeT<MemSpace::host> const*            host_world_   = {nullptr};
    PlacedVolumeT<MemSpace::device> const*          device_world_ = {nullptr};
    std::shared_ptr<const celeritas::VecgeomParams> geom_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(G4VecgeomConvTest, conversion)
{
    // access geometry through the VecGeom GeoManager class
    auto& geomgr = vecgeom::GeoManager::Instance();
    EXPECT_TRUE(&geomgr);
    EXPECT_TRUE(geomgr.IsClosed());
    EXPECT_EQ(4, geomgr.getMaxDepth());

    // PlacedVolume-related tests
    const auto& pworld = geomgr.GetWorld();
    EXPECT_TRUE(pworld);
    EXPECT_TRUE(pworld->GetLogicalVolume());
    EXPECT_LT(0, pworld->GetDaughters().size());

    // pworld->PrintContent();
}

//---------------------------------------------------------------------------//

TEST_F(G4VecgeomConvTest, host_world)
{
    // access world volumes through GeoParams object
    const auto& geom = *this->geom_.get();
    EXPECT_EQ(4, geom.num_volumes());
    EXPECT_EQ(4, geom.max_depth());

    // const auto& world = geom.host_ref();
    //  get uniform pointer to PlacedVolume
    // PlacedVolumeT<MemSpace::host> const& world{*pworld};
    EXPECT_EQ("World", geom.id_to_label(VolumeId{0}));
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{1}));
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{2}));
    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{3}));
}

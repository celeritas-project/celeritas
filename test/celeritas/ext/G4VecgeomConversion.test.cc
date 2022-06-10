//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/VecgeomData.hh"
#include "celeritas/ext/detail/G4VecgeomConverter.hh"
#include "celeritas/geo/GeoParams.hh"

#include "Geant4/G4GDMLParser.hh"
#include "Geant4/G4GeometryManager.hh"
#include "Geant4/G4PhysicalVolumeStore.hh"
#include "celeritas_test.hh"

using namespace celeritas;
template<MemSpace M>
using PlacedVolumeT = typename detail::VecgeomTraits<M>::PlacedVolume;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The \e four-levels.gdml file is used to construct a Geant4 model,
 * which is then converted into a corresponding Vecgeom model.
 */
class G4VecgeomConvTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        CELER_LOG(info) << "G4VecgeomConvTest::SetUp()";
        // GeantSetup setup = {"four_levels.gdml", GeantSetupOptions{}};
        G4GDMLParser parser;
        std::string  filename
            = ::test::Test::test_data_path("celeritas", "four-levels.gdml");
        // parser.SetOverlapCheck(true);
        bool validate_xml = false;
        parser.Read(filename.c_str(), validate_xml);

        // Convert Geant4 model into VecGeom model
        geom_ = std::make_shared<GeoParams>(parser.GetWorldVolume());

        // Tried to delete G4 geometry - it crashes at 2nd test
        // auto* g4_volume_store = G4PhysicalVolumeStore::GetInstance();
        // const auto& volume_map = g4_volume_store->GetMap(); // Not all G4
        // versions support this CELER_EXPECT(volume_map.size() > 0);
        // G4GeometryManager::GetInstance()->OpenGeometry();
        // G4PhysicalVolumeStore::GetInstance()->Clean();
        // CELER_ASSERT(volume_map.size() == 0);
    }

    void TearDown() override
    {
        CELER_LOG(info) << "G4VecgeomConvTest::TearDown()";
        geom_.reset();
        CELER_ASSERT(!vecgeom::GeoManager::Instance().GetWorld());
    }

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

TEST_F(G4VecgeomConvTest, params_access)
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

//---------------------------------------------------------------------------//

TEST_F(G4VecgeomConvTest, host_access_vecgeom)
{
    // access world volumes through GeoParams object
    const auto& geom = *this->geom_.get();

    // const auto& world = geom.host_ref();
    //  get uniform pointer to PlacedVolume
    // PlacedVolumeT<MemSpace::host> const& world{*pworld};
    const auto& geoparams = geom.host_ref();
    EXPECT_TRUE(geoparams);

    // address to VecGeom objects
    const auto& world = *geoparams.world_volume;
    EXPECT_TRUE(world.GetLogicalVolume());
    EXPECT_EQ(8, world.GetDaughters().size());

    // world->PrintContent();
}

//---------------------------------------------------------------------------//

#include "corecel/data/CollectionStateStore.hh"

#include "G4VecgeomConversion.test.hh" // for definition of VGGTestInput,Output structs

// Since VecGeom is currently CUDA-only, we cannot use the TEST_IF_CELER_DEVICE
// macro (which also allows HIP).
#if CELERITAS_USE_CUDA
#    define TEST_IF_CELERITAS_CUDA(name) name
#else
#    define TEST_IF_CELERITAS_CUDA(name) DISABLED_##name
#endif

TEST_F(G4VecgeomConvTest, TEST_IF_CELERITAS_CUDA(device_tracking))
{
    using StateStore = CollectionStateStore<VecgeomStateData, MemSpace::device>;

    // Set up test input
    // TODO: apply mm -> cm conversion to volume dimensions in the G4->VecGeom
    // conversion
    celeritas_test::G4VGConvTestInput input;
    input.init = {{{100, 100, 100}, {1, 0, 0}},
                  {{100, 100, -100}, {1, 0, 0}},
                  {{100, -100, 100}, {1, 0, 0}},
                  {{100, -100, -100}, {1, 0, 0}},
                  {{-100, 100, 100}, {-1, 0, 0}},
                  {{-100, 100, -100}, {-1, 0, 0}},
                  {{-100, -100, 100}, {-1, 0, 0}},
                  {{-100, -100, -100}, {-1, 0, 0}}};

    const auto& geoparams = *this->geom_.get();
    StateStore  device_states(geoparams.host_ref(), input.init.size());
    input.max_segments = 4;
    input.params       = geoparams.device_ref();
    input.state        = device_states.ref();

    // Run kernel
    auto output = g4vgconv_test(input);

    static const int expected_ids[] = {2,  1,  0, -1, 2,  1,  0, -1, 2,  1, 0,
                                       -1, 2,  1, 0,  -1, 2,  1, 0,  -1, 2, 1,
                                       0,  -1, 2, 1,  0,  -1, 2, 1,  0,  -1};

    static const double expected_distances[]
        = {50, 10, 10, 70, 50, 10, 10, 70, 50, 10, 10, 70, 50, 10, 10, 70,
           50, 10, 10, 70, 50, 10, 10, 70, 50, 10, 10, 70, 50, 10, 10, 70};

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(expected_distances, output.distances);
}

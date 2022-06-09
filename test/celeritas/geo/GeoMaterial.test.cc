//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/GlobalTestBase.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/io/ImportData.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTest : public celeritas_test::GlobalGeoTestBase
{
    const char* geometry_basename() const override { return "simple-cms"; }

    SPConstParticle build_particle() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCutoff   build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics  build_physics() override { CELER_ASSERT_UNREACHABLE(); }

    SPConstMaterial build_material() override
    {
        return MaterialParams::from_import(data_);
    }

    SPConstGeoMaterial build_geomaterial() override
    {
        // Create geometry/material coupling
        GeoMaterialParams::Input input;
        input.geometry  = this->geometry();
        input.materials = this->material();
        input.volume_to_mat.resize(data_.volumes.size());
        input.volume_names.resize(data_.volumes.size());

        for (const auto vol_idx : range(data_.volumes.size()))
        {
            const ImportVolume& volume = data_.volumes[vol_idx];

            input.volume_to_mat[vol_idx] = MaterialId{volume.material_id};
            input.volume_names[vol_idx]  = volume.name;
        }
        return std::make_shared<GeoMaterialParams>(std::move(input));
    }

    void SetUp() override
    {
        // Load ROOT file
        // The simple-cms.root has no ImportData processes stored. Process data
        // is not used and increases the file size by > 5x.
        std::string root_file
            = this->test_data_path("celeritas", "simple-cms.root");
        data_ = RootImporter(root_file.c_str())();
    }

  private:
    ImportData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GeoMaterialTest, host)
{
    // Geometry track view and mat view
    const auto& geo_params = *this->geometry();
    const auto& mat_params = *this->material();
    CollectionStateStore<GeoStateData, MemSpace::host> geo_state(geo_params, 1);
    GeoTrackView    geo(geo_params.host_ref(), geo_state.ref(), ThreadId{0});
    GeoMaterialView geo_mat_view(this->geomaterial()->host_ref());

    // Track across layers to get a truly implementation-independent
    // comparison of material IDs encountered.
    std::vector<std::string> materials;

    geo = {{0, 0, 0}, {1, 0, 0}};
    while (!geo.is_outside())
    {
        MaterialId matid = geo_mat_view.material_id(geo.volume_id());

        materials.push_back(matid ? mat_params.id_to_label(matid)
                                  : "[invalid]");

        geo.find_next_step();
        geo.move_to_boundary();
        geo.cross_boundary();
    }

    // PRINT_EXPECTED(materials);
    static const std::string expected_materials[]
        = {"vacuum", "Si", "Pb", "C", "Ti", "Fe", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

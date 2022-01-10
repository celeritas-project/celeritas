//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "geometry/GeoMaterialParams.hh"

#include "base/CollectionStateStore.hh"
#include "geometry/GeoData.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "io/RootImporter.hh"
#include "io/ImportData.hh"

#include "celeritas_test.hh"
#include "GeoTestBase.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTest : public celeritas_test::GeoTestBase<celeritas::GeoParams>
{
    const char* dirname() const override { return "geometry"; }
    const char* filebase() const override { return "simple-cms"; }

    void SetUp() override
    {
        // Load ROOT file
        std::string root_file
            = this->test_data_path("geometry", "simple-cms.root");
        auto data = RootImporter(root_file.c_str())();

        // Set up shared material data
        material_ = MaterialParams::from_import(data);

        // Create geometry/material coupling
        GeoMaterialParams::Input input;
        input.geometry  = this->geometry();
        input.materials = material_;
        input.volume_to_mat.resize(data.volumes.size());
        input.volume_names.resize(data.volumes.size());

        for (const auto vol_idx : range(data.volumes.size()))
        {
            const ImportVolume& volume = data.volumes[vol_idx];

            input.volume_to_mat[vol_idx] = MaterialId{volume.material_id};
            input.volume_names[vol_idx]  = volume.name;
        }
        geo_mat_ = std::make_shared<GeoMaterialParams>(std::move(input));
    }

  protected:
    std::shared_ptr<const MaterialParams>    material_;
    std::shared_ptr<const GeoMaterialParams> geo_mat_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GeoMaterialTest, host)
{
    // Geometry track view and mat view
    const auto& geo_params = *this->geometry();
    CollectionStateStore<GeoStateData, MemSpace::host> geo_state(geo_params, 1);
    GeoTrackView    geo(geo_params.host_ref(), geo_state.ref(), ThreadId{0});
    GeoMaterialView geo_mat_view(geo_mat_->host_ref());

    // Track across layers to get a truly implementation-independent
    // comparison of material IDs encountered.
    std::vector<std::string> materials;

    geo = {{0, 0, 0}, {1, 0, 0}};
    while (!geo.is_outside())
    {
        MaterialId matid = geo_mat_view.material_id(geo.volume_id());

        materials.push_back(matid ? material_->id_to_label(matid)
                                  : "[invalid]");

        geo.find_next_step();
        geo.move_across_boundary();
    }

    // PRINT_EXPECTED(materials);
    static const std::string expected_materials[]
        = {"vacuum", "Si", "Pb", "C", "Ti", "Fe", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

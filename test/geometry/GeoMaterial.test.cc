//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "geometry/GeoMaterialParams.hh"

#include "geometry/GeoParams.hh"
#include "geometry/GeoMaterialView.hh"
#include "io/RootImporter.hh"
#include "io/ImportData.hh"

#include "GeoTestBase.hh"
#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTest : public celeritas_test::GeoTestBase
{
    std::string filename() const override { return "slabsGeometry.gdml"; }

    void SetUp() override
    {
        // Load ROOT file
        std::string root_file
            = this->test_data_path("io", "geant-exporter-data.root");
        const auto data
            = RootImporter(root_file.c_str())("geant4_data", "ImportData");

        // Set up shared material data
        material_ = std::move(MaterialParams::from_import(data));

        // Create geometry/material coupling
        GeoMaterialParams::Input input;
        input.geometry  = this->geo_params();
        input.materials = material_;
        input.volume_to_mat
            = std::vector<MaterialId>(input.geometry->num_volumes());
        for (const auto& kv : data.geometry.volid_to_matid_map())
        {
            CELER_ASSERT(kv.first < input.volume_to_mat.size());
            CELER_ASSERT(kv.second < input.materials->num_materials());
            input.volume_to_mat[kv.first] = MaterialId{kv.second};
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
    const unsigned int expected_mat_id[] = {1, 1, 1, 1, 0};

    const auto& geo = *this->geo_params();
    ;
    EXPECT_EQ(5, geo.num_volumes());
    for (auto i : range(geo.num_volumes()))
    {
        GeoMaterialView geo_mat_view(geo_mat_->host_pointers(), VolumeId{i});
        EXPECT_EQ(MaterialId{expected_mat_id[i]}, geo_mat_view.material());
    }
}

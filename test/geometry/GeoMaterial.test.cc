//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "geometry/GeoMaterialParams.hh"

#include "celeritas_test.hh"
#include "geometry/GeoMaterialView.hh"
#include "io/RootLoader.hh"
#include "io/MaterialParamsLoader.hh"

#include "GeoParamsTest.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTest : public celeritas::Test
{
    void SetUp() override
    {
        // Load ROOT file
        std::string root_file
            = this->test_data_path("io", "geant-exporter-data.root");
        auto root_loader = RootLoader(root_file.c_str());

        // Set up shared material data
        material_ = std::move(MaterialParamsLoader(root_loader)());

        // Set up shared geometry data
        std::string geo_file
            = this->test_data_path("geometry", "four-steel-slabs.gdml");
        geometry_ = std::make_shared<GeoParams>(geo_file.c_str());

        // Create geometry/material coupling
        GeoMaterialParams::Input input;
        input.geometry  = geometry_;
        input.materials = material_;
        input.volume_to_mat
            = std::vector<MaterialId>(input.geometry->num_volumes());
        for (const auto& kv : loaded.geometry->volid_to_matid_map())
        {
            CELER_ASSERT(kv.first < input.volume_to_mat.size());
            CELER_ASSERT(kv.second < input.materials->num_materials());
            input.volume_to_mat[kv.first] = MaterialId{kv.second};
        }
        geo_mat_ = std::make_shared<GeoMaterialParams>(std::move(input));
    }

  protected:
    std::shared_ptr<const GeoParams>         geometry_;
    std::shared_ptr<const MaterialParams>    material_;
    std::shared_ptr<const GeoMaterialParams> geo_mat_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GeoMaterialTest, all)
{
    const unsigned int expected_mat_id[] = {1, 1, 1, 1, 0};

    EXPECT_EQ(5, geometry_->num_volumes());
    for (auto i : range(geometry_->num_volumes()))
    {
        GeoMaterialView geo_mat_view(geo_mat_->host_pointers(), VolumeId{i});
        EXPECT_EQ(MaterialId{expected_mat_id[i]}, geo_mat_view.material());
    }
}

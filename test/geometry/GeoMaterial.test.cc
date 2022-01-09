//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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

#include "celeritas_test.hh"
#include "GeoTestBase.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTest : public celeritas_test::GeoTestBase<celeritas::GeoParams>
{
    const char* dirname() const override { return "geometry"; }
    const char* filebase() const override { return "four-steel-slabs"; }

    void SetUp() override
    {
        // Load ROOT file
        std::string root_file
            = this->test_data_path("io", "geant-exporter-data.root");
        const auto data = RootImporter(root_file.c_str())();

        // Set up shared material data
        material_ = MaterialParams::from_import(data);

        // Create geometry/material coupling
        GeoMaterialParams::Input input;
        input.geometry  = this->geometry();
        input.materials = material_;
        input.volume_to_mat
            = std::vector<MaterialId>(input.geometry->num_volumes());

        CELER_ASSERT(data.volumes.size() == input.volume_to_mat.size());
        CELER_ASSERT(data.materials.size() == input.materials->num_materials());

        for (const auto volume_id : range(data.volumes.size()))
        {
            const auto& volume             = data.volumes.at(volume_id);
            input.volume_to_mat[volume_id] = MaterialId{volume.material_id};
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

    const auto& geo = *this->geometry();
    EXPECT_EQ(5, geo.num_volumes());

    GeoMaterialView geo_mat_view(geo_mat_->host_ref());
    for (auto i : range(geo.num_volumes()))
    {
        EXPECT_EQ(MaterialId{expected_mat_id[i]},
                  geo_mat_view.material_id(VolumeId{i}));
    }
}

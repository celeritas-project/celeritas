//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.test.cc
//---------------------------------------------------------------------------//
#include "geometry/GeoParamsTest.hh"

#include "celeritas_config.h"
#include "comm/Device.hh"
#if CELERITAS_USE_CUDA
#    include <VecGeom/management/CudaManager.h>
#endif

using celeritas::VolumeId;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class GeoParamsHostTest : public GeoParamsTest
{
  public:
    void SetUp() override
    {
        host_view = this->params()->host_pointers();
        CELER_ASSERT(host_view.world_volume);
    }

    // Views
    celeritas::GeoParamsPointers host_view;
};

//---------------------------------------------------------------------------//

TEST_F(GeoParamsHostTest, accessors)
{
    const auto& geom = *(this->params());
    EXPECT_EQ(11, geom.num_volumes());
    EXPECT_EQ(4, geom.max_depth());

    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{0}) );
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{1}) );
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{2}) );

    unsigned int nvols = geom.num_volumes();
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{nvols - 2}));
    EXPECT_EQ("World", geom.id_to_label(VolumeId{nvols - 1}));
}

//---------------------------------------------------------------------------//

TEST_F(GeoParamsHostTest, print_geometry)
{
    if (!celeritas::device())
    {
        SKIP("CUDA is disabled");
    }

    // Geometry functionality
    auto device_view = this->params()->device_pointers();
    EXPECT_NE(nullptr, device_view.world_volume);

#if CELERITAS_USE_CUDA
    // debug: print geometry information from device
    //vecgeom::cxx::CudaManager::Instance().PrintGeometry();
#endif
}

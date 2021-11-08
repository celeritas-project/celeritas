//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OrangeGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "OrangeGeoTestBase.hh"

#include <fstream>
#include "celeritas_config.h"

#include "orange/construct/SurfaceInput.hh"
#include "orange/construct/SurfaceInserter.hh"
#include "orange/construct/VolumeInput.hh"
#include "orange/construct/VolumeInserter.hh"
#if CELERITAS_USE_JSON
#    include "orange/construct/SurfaceInputIO.json.hh"
#    include "orange/construct/VolumeInputIO.json.hh"
#endif
#include "orange/surfaces/Sphere.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Load a geometry from the given JSON filename.
 */
void OrangeGeoTestBase::build_geometry(const char* filename)
{
    CELER_EXPECT(!params_);
    CELER_EXPECT(filename);
    CELER_VALIDATE(CELERITAS_USE_JSON,
                   << "JSON is not enabled so geometry cannot be loaded");
#if CELERITAS_USE_JSON
    std::ifstream infile(
        this->test_data_path("orange", "five-volumes.org.json"));
    CELER_VALIDATE(infile, << "failed to open geometry path");

    auto        full_inp  = nlohmann::json::parse(infile);
    const auto& universes = full_inp["universes"];

    CELER_VALIDATE(universes.size() == 1,
                   << "input geometry has " << universes.size()
                   << "universes; at present there must be a single global "
                      "universe");

    OrangeParamsData<Ownership::value, MemSpace::host> host_data;

    {
        // Insert surfaces
        SurfaceInserter insert(&host_data.surfaces);
        insert(universes[0]["surfaces"].get<SurfaceInput>());
    }

    {
        // Insert volumes
        VolumeInserter insert(&host_data.volumes);
        for (const auto& vol_inp : universes[0]["cells"])
        {
            insert(vol_inp.get<VolumeInput>());
        }
    }

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    params_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};

#endif
    CELER_ENSURE(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry with one infinite volume.
 */
void OrangeGeoTestBase::build_geometry(OneVolInput)
{
    CELER_EXPECT(!params_);
    OrangeParamsData<Ownership::value, MemSpace::host> host_data;

    {
        // Insert volumes
        VolumeInserter insert(&host_data.volumes);
        VolumeInput    inp;
        inp.logic = {logic::ltrue};
        insert(inp);
    }

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    params_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};
    CELER_ENSURE(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry with one infinite volume.
 */
void OrangeGeoTestBase::build_geometry(TwoVolInput inp)
{
    CELER_EXPECT(!params_);
    CELER_EXPECT(inp.radius > 0);
    OrangeParamsData<Ownership::value, MemSpace::host> host_data;

    {
        // Insert surfaces
        SurfaceInserter insert(&host_data.surfaces);
        insert(Sphere({0, 0, 0}, inp.radius));
    }

    {
        // Insert volumes
        VolumeInserter insert(&host_data.volumes);
        {
            VolumeInput inp;
            // Inside
            inp.logic             = {0, logic::lnot};
            inp.faces             = {SurfaceId{0}};
            inp.num_intersections = 2;
            insert(inp);

            // Outside
            inp.logic = {0};
            insert(inp);
        }
    }

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    params_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};
    CELER_ENSURE(params_);
}

//---------------------------------------------------------------------------//
//! Default destructor
OrangeGeoTestBase::~OrangeGeoTestBase() = default;

//---------------------------------------------------------------------------//
} // namespace celeritas_test

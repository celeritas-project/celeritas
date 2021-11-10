//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OrangeGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "OrangeGeoTestBase.hh"

#include <fstream>
#include <sstream>
#include <utility>
#include "celeritas_config.h"

#include "base/Join.hh"
#include "orange/Types.hh"
#include "orange/construct/SurfaceInput.hh"
#include "orange/construct/SurfaceInserter.hh"
#include "orange/construct/VolumeInput.hh"
#include "orange/construct/VolumeInserter.hh"
#if CELERITAS_USE_JSON
#    include "orange/construct/SurfaceInputIO.json.hh"
#    include "orange/construct/VolumeInputIO.json.hh"
#endif
#include "orange/surfaces/Sphere.hh"
#include "orange/surfaces/SurfaceAction.hh"
#include "orange/surfaces/SurfaceIO.hh"
#include "orange/universes/VolumeView.hh"

using namespace celeritas;

namespace celeritas_test
{
namespace
{
struct ToStream
{
    std::ostream& os;

    template<class S>
    std::ostream& operator()(S&& surf) const
    {
        os << surf;
        return os;
    }
};

} // namespace
//---------------------------------------------------------------------------//
/*!
 * Convert a vector of senses to a string.
 */
std::string OrangeGeoTestBase::senses_to_string(Span<const Sense> senses)
{
    std::ostringstream os;
    os << '{' << celeritas::join(senses.begin(), senses.end(), ' ', [](Sense s) {
        return to_char(s);
    }) << '}';
    return os.str();
}

//---------------------------------------------------------------------------//
//! Default destructor
OrangeGeoTestBase::~OrangeGeoTestBase() = default;

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

    OrangeParamsData<Ownership::value, MemSpace::host> host_data;

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
#endif

    return this->build_impl(std::move(host_data));
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

    return this->build_impl(std::move(host_data));
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

    return this->build_impl(std::move(host_data));
}

//---------------------------------------------------------------------------//
/*!
 * Print geometry description.
 */
void OrangeGeoTestBase::describe(std::ostream& os) const
{
    CELER_EXPECT(params_);

    os << "# Surfaces\n";
    Surfaces surfaces(this->params_host_ref().surfaces);
    auto     surf_to_stream = make_surface_action(surfaces, ToStream{os});

    // Loop over all surfaces and apply
    for (auto id : range(SurfaceId{surfaces.num_surfaces()}))
    {
        os << " - " << id.get() << ": ";
        surf_to_stream(id);
        os << '\n';
    }
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Update and save geometry.
 */
void OrangeGeoTestBase::build_impl(ParamsHostValue&& host_data)
{
    CELER_EXPECT(host_data);

    // Calculate max faces
    size_type max_faces         = 0;
    size_type max_intersections = 0;
    for (auto vol_id : range(VolumeId{host_data.volumes.size()}))
    {
        const VolumeDef& def = host_data.volumes.defs[vol_id];
        max_faces = std::max<size_type>(max_faces, def.faces.size());
        max_intersections
            = std::max<size_type>(max_intersections, def.num_intersections);
    }
    host_data.scalars.max_faces         = max_faces;
    host_data.scalars.max_intersections = max_intersections;

    sense_storage_.resize(max_faces);

    // Construct device values and device/host references
    params_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};
    CELER_ENSURE(params_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test

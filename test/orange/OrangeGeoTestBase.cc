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
#include "orange/surfaces/Sphere.hh"
#include "orange/surfaces/SurfaceAction.hh"
#include "orange/surfaces/SurfaceIO.hh"
#include "orange/universes/VolumeView.hh"

#if CELERITAS_USE_JSON
#    include "base/Array.json.hh"
#    include "orange/construct/SurfaceInputIO.json.hh"
#    include "orange/construct/VolumeInputIO.json.hh"
#endif

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
    const auto& uni = universes[0];

    {
        // Insert surfaces
        SurfaceInserter insert(&host_data.surfaces);
        insert(uni["surfaces"].get<SurfaceInput>());
        uni["surface_names"].get_to(surf_names_);
    }

    {
        // Insert volumes
        VolumeInserter insert(&host_data.volumes);
        for (const auto& vol_inp : uni["cells"])
        {
            insert(vol_inp.get<VolumeInput>());
        }
        uni["cell_names"].get_to(vol_names_);
    }

    {
        // Save bbox
        const auto& bbox = uni["bbox"];
        bbox[0].get_to(bbox_lower_);
        bbox[1].get_to(bbox_upper_);
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
        // No suerfaces
        surf_names_ = {};
    }
    {
        // Insert volumes
        VolumeInserter insert(&host_data.volumes);
        VolumeInput    inp;
        inp.logic = {logic::ltrue};
        insert(inp);
        vol_names_ = {"infinite"};
    }

    // Save fake bbox for sampling
    bbox_lower_ = {-0.5, -0.5, -0.5};
    bbox_upper_ = {0.5, 0.5, 0.5};

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
        surf_names_ = {"sphere"};
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
        vol_names_ = {"inside", "outside"};
    }

    // Save bbox
    bbox_lower_ = {-inp.radius, -inp.radius, -inp.radius};
    bbox_upper_ = {inp.radius, inp.radius, inp.radius};

    return this->build_impl(std::move(host_data));
}

//---------------------------------------------------------------------------//
/*!
 * Print geometry description.
 *
 * This is just developer-oriented code until we get the full ORANGE metadata
 * ported.
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
        os << " - " << surf_names_[id.get()] << "(" << id.get() << "): ";
        surf_to_stream(id);
        os << '\n';
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the surface from its label (nullptr allowed)
 */
SurfaceId OrangeGeoTestBase::find_surface(const char* label) const
{
    SurfaceId surface_id;
    if (label)
    {
        auto iter = surf_ids_.find(label);
        CELER_VALIDATE(iter != surf_ids_.end(),
                       << "nonexistent surface label '" << label << '\'');
        surface_id = iter->second;
    }
    return surface_id;
}

//---------------------------------------------------------------------------//
/*!
 * Find the volume from its label (nullptr allowed)
 */
VolumeId OrangeGeoTestBase::find_volume(const char* label) const
{
    VolumeId volume_id;
    if (label)
    {
        auto iter = vol_ids_.find(label);
        CELER_VALIDATE(iter != vol_ids_.end(),
                       << "nonexistent volume label '" << label << '\'');
        volume_id = iter->second;
    }
    return volume_id;
}

//---------------------------------------------------------------------------//
/*!
 * Surface name (or sentinel if no surface).
 */
std::string OrangeGeoTestBase::id_to_label(SurfaceId surf) const
{
    CELER_EXPECT(!surf || surf < surf_names_.size());
    if (!surf)
        return "[none]";

    return surf_names_[surf.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Volume name (or sentinel if no volume).
 */
std::string OrangeGeoTestBase::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(!vol || vol < vol_names_.size());
    if (!vol)
        return "[none]";

    return vol_names_[vol.get()];
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Update and save geometry.
 */
void OrangeGeoTestBase::build_impl(ParamsHostValue&& host_data)
{
    CELER_EXPECT(host_data.surfaces && host_data.volumes);

    // Calculate max faces and intersections, reserving at least one to
    // improve error checking in state
    size_type max_faces         = 1;
    size_type max_intersections = 1;
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
    CELER_ASSERT(host_data);
    params_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};

    // Build id/surface mapping
    for (auto vid : range(VolumeId(vol_names_.size())))
    {
        auto iter_inserted = vol_ids_.insert({vol_names_[vid.get()], vid});
        CELER_VALIDATE(iter_inserted.second,
                       << "duplicate volume name '"
                       << iter_inserted.first->first << '\'');
    }
    for (auto sid : range(SurfaceId(surf_names_.size())))
    {
        auto iter_inserted = surf_ids_.insert({surf_names_[sid.get()], sid});
        CELER_VALIDATE(iter_inserted.second,
                       << "duplicate surface name '"
                       << iter_inserted.first->first << '\'');
    }

    CELER_ENSURE(params_);
    CELER_ENSURE(surf_names_.size() == this->params_host_ref().surfaces.size());
    CELER_ENSURE(vol_names_.size() == this->params_host_ref().volumes.size());
    CELER_ENSURE(surf_ids_.size() == surf_names_.size());
    CELER_ENSURE(vol_ids_.size() == vol_names_.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test

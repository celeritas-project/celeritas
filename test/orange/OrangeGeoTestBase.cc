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

    params_
        = std::make_unique<Params>(this->test_data_path("orange", filename));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry with one infinite volume.
 */
void OrangeGeoTestBase::build_geometry(OneVolInput inp)
{
    CELER_EXPECT(!params_);
    Params::Input input;
    {
        // Insert volumes
        VolumeInserter insert(&input.volumes);
        VolumeInput    vi;
        vi.logic = {logic::ltrue};
        vi.flags = (inp.complex_tracking ? VolumeInput::Flags::internal_surfaces
                                         : 0);
        insert(vi);
        input.volume_labels = {"infinite"};
    }

    // Save fake bbox for sampling
    input.bbox = {{-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}};

    params_ = std::make_unique<Params>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry with one infinite volume.
 */
void OrangeGeoTestBase::build_geometry(TwoVolInput inp)
{
    CELER_EXPECT(!params_);
    CELER_EXPECT(inp.radius > 0);
    Params::Input input;

    {
        // Insert surfaces
        SurfaceInserter insert(&input.surfaces);
        insert(Sphere({0, 0, 0}, inp.radius));
        input.surface_labels = {"sphere"};
    }
    {
        // Insert volumes
        VolumeInserter insert(&input.volumes);
        {
            VolumeInput vi;
            vi.faces             = {SurfaceId{0}};
            vi.num_intersections = 2;

            // Outside
            vi.logic = {0};
            insert(vi);

            // Inside
            vi.logic = {0, logic::lnot};
            insert(vi);
        }
        input.volume_labels = {"outside", "inside"};
    }

    // Save bbox
    input.bbox = {{-inp.radius, -inp.radius, -inp.radius},
                  {inp.radius, inp.radius, inp.radius}};

    params_ = std::make_unique<Params>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Lazily create and get a single-serving host state.
 */
auto OrangeGeoTestBase::host_state() -> const HostStateRef&
{
    CELER_EXPECT(params_);
    if (!host_state_)
    {
        host_state_ = HostStateStore(this->params(), 1);
    }
    return host_state_.ref();
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
    Surfaces surfaces(this->params().host_ref().surfaces);
    auto     surf_to_stream = make_surface_action(surfaces, ToStream{os});

    // Loop over all surfaces and apply
    for (auto id : range(SurfaceId{surfaces.num_surfaces()}))
    {
        os << " - " << params_->id_to_label(id) << "(" << id.get() << "): ";
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
    CELER_EXPECT(params_);
    SurfaceId surface_id;
    if (label)
    {
        surface_id = params_->find_surface(label);
        CELER_VALIDATE(surface_id,
                       << "nonexistent surface label '" << label << '\'');
    }
    return surface_id;
}

//---------------------------------------------------------------------------//
/*!
 * Find the volume from its label (nullptr allowed)
 */
VolumeId OrangeGeoTestBase::find_volume(const char* label) const
{
    CELER_EXPECT(params_);
    VolumeId volume_id;
    if (label)
    {
        volume_id = params_->find_volume(label);
        CELER_VALIDATE(volume_id,
                       << "nonexistent volume label '" << label << '\'');
    }
    return volume_id;
}

//---------------------------------------------------------------------------//
/*!
 * Surface name (or sentinel if no surface).
 */
std::string OrangeGeoTestBase::id_to_label(SurfaceId surf) const
{
    if (!surf)
        return "[none]";

    return params_->id_to_label(surf);
}

//---------------------------------------------------------------------------//
/*!
 * Volume name (or sentinel if no volume).
 */
std::string OrangeGeoTestBase::id_to_label(VolumeId vol) const
{
    if (!vol)
        return "[none]";

    return params_->id_to_label(vol);
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test

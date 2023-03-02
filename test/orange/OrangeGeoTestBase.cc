//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "OrangeGeoTestBase.hh"

#include <fstream>
#include <sstream>
#include <utility>

#include "celeritas_config.h"
#include "corecel/data/Ref.hh"
#include "corecel/io/Join.hh"
#include "orange/Types.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/construct/SurfaceInputBuilder.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SurfaceAction.hh"
#include "orange/surf/SurfaceIO.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
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

OrangeInput to_input(UnitInput u)
{
    OrangeInput result;
    result.units.push_back(std::move(u));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Convert a vector of senses to a string.
 */
std::string OrangeGeoTestBase::senses_to_string(Span<Sense const> senses)
{
    std::ostringstream os;
    os << '{' << join(senses.begin(), senses.end(), ' ', [](Sense s) {
        return to_char(s);
    }) << '}';
    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Convert a string to a vector of senses.
 */
std::vector<Sense> OrangeGeoTestBase::string_to_senses(std::string const& s)
{
    std::vector<Sense> result(s.size());
    std::transform(s.begin(), s.end(), result.begin(), [](char c) {
        CELER_EXPECT(c == '+' || c == '-');
        return c == '+' ? Sense::outside : Sense::inside;
    });
    return result;
}

//---------------------------------------------------------------------------//
//! Default destructor
OrangeGeoTestBase::~OrangeGeoTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Load a geometry from the given JSON filename.
 */
void OrangeGeoTestBase::build_geometry(char const* filename)
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
    UnitInput input;
    {
        // Insert volumes
        VolumeInput vi;
        vi.logic = {logic::ltrue};
        vi.flags = (inp.complex_tracking ? VolumeInput::Flags::internal_surfaces
                                         : 0);
        vi.label = "infinite";
        input.volumes.push_back(std::move(vi));
    }

    // Save fake bbox for sampling
    input.bbox = {{-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}};

    input.label = "one volume";

    return this->build_geometry(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry with one infinite volume.
 */
void OrangeGeoTestBase::build_geometry(TwoVolInput inp)
{
    CELER_EXPECT(!params_);
    CELER_EXPECT(inp.radius > 0);
    UnitInput input;

    {
        // Insert surfaces
        SurfaceInputBuilder insert(&input.surfaces);
        insert(Sphere({0, 0, 0}, inp.radius), Label("sphere"));
    }
    {
        // Insert volumes
        VolumeInput vi;
        vi.faces = {LocalSurfaceId{0}};

        // Outside
        vi.logic = {0};
        vi.label = "outside";
        input.volumes.push_back(vi);

        // Inside
        vi.logic = {0, logic::lnot};
        vi.label = "inside";
        input.volumes.push_back(vi);
    }

    // Save bbox
    input.bbox = {{-inp.radius, -inp.radius, -inp.radius},
                  {inp.radius, inp.radius, inp.radius}};

    input.label = "two volumes";

    return this->build_geometry(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry from a single global unit.
 */
void OrangeGeoTestBase::build_geometry(UnitInput input)
{
    CELER_EXPECT(input);
    params_ = std::make_unique<Params>(to_input(std::move(input)));
}

//---------------------------------------------------------------------------//
/*!
 * Lazily create and get a single-serving host state.
 */
auto OrangeGeoTestBase::host_state() -> HostStateRef const&
{
    CELER_EXPECT(params_);
    if (!host_state_)
    {
        host_state_ = HostStateStore(this->params().host_ref(), 1);
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

    // TODO: update when multiple units are in play
    auto const& host_ref = this->params().host_ref();
    CELER_ASSERT(host_ref.simple_unit.size() == 1);

    os << "# Surfaces\n";
    Surfaces surfaces(host_ref, host_ref.simple_unit[SimpleUnitId{0}].surfaces);
    auto surf_to_stream = make_surface_action(surfaces, ToStream{os});

    // Loop over all surfaces and apply
    for (auto id : range(LocalSurfaceId{surfaces.num_surfaces()}))
    {
        os << " - " << this->id_to_label(UniverseId{0}, id) << "(" << id.get()
           << "): ";
        surf_to_stream(id);
        os << '\n';
    }
}

//---------------------------------------------------------------------------//
/*!
 * Find the surface from its label (nullptr allowed)
 */
SurfaceId OrangeGeoTestBase::find_surface(char const* label) const
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
VolumeId OrangeGeoTestBase::find_volume(char const* label) const
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
std::string
OrangeGeoTestBase::id_to_label(UniverseId uid, LocalSurfaceId surfid) const
{
    if (!surfid)
        return "[none]";

    detail::UnitIndexer ui(this->params().host_ref().unit_indexer_data);
    return params_->id_to_label(ui.global_surface(uid, surfid)).name;
}

//---------------------------------------------------------------------------//
/*!
 * Surface name (or sentinel if no surface) within UniverseId{0}.
 */
std::string OrangeGeoTestBase::id_to_label(LocalSurfaceId surfid) const
{
    return this->id_to_label(UniverseId{0}, surfid);
}

//---------------------------------------------------------------------------//
/*!
 * Volume name (or sentinel if no volume).
 */
std::string
OrangeGeoTestBase::id_to_label(UniverseId uid, LocalVolumeId volid) const
{
    if (!volid)
        return "[none]";

    detail::UnitIndexer ui(this->params().host_ref().unit_indexer_data);
    return params_->id_to_label(ui.global_volume(uid, volid)).name;
}

//---------------------------------------------------------------------------//
/*!
 * Volume name (or sentinel if no volume) within UniverseId{0}.
 */
std::string OrangeGeoTestBase::id_to_label(LocalVolumeId volid) const
{
    return this->id_to_label(UniverseId{0}, volid);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

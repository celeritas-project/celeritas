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
#include "corecel/ScopedLogStorer.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "orange/OrangeParams.hh"
#include "orange/Types.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/detail/UniverseIndexer.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SurfaceIO.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
OrangeInput to_input(UnitInput u)
{
    OrangeInput result;
    result.universes.push_back(std::move(u));
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
/*!
 * Load a geometry from the given JSON filename.
 */
void OrangeGeoTestBase::build_geometry(char const* filename)
{
    CELER_EXPECT(!params_);
    CELER_EXPECT(filename);
    CELER_VALIDATE(CELERITAS_USE_JSON,
                   << "JSON is not enabled so geometry cannot be loaded");

    ScopedLogStorer scoped_log_{&celeritas::world_logger()};
    params_
        = std::make_unique<Params>(this->test_data_path("orange", filename));

    static char const* const expected_log_levels[] = {"info"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a geometry with one infinite volume.
 */
void OrangeGeoTestBase::build_geometry(OneVolInput inp)
{
    CELER_EXPECT(!params_);
    UnitInput input;
    input.label = "one volume";
    // Fake bbox for sampling
    input.bbox = {{-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}};

    input.volumes = {[&inp] {
        VolumeInput vi;
        vi.logic = {logic::ltrue};
        vi.flags = (inp.complex_tracking ? static_cast<logic_int>(
                        VolumeInput::Flags::internal_surfaces)
                                         : 0);
        vi.zorder = ZOrder::media;
        vi.label = "infinite";
        return vi;
    }()};

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
    input.label = "two volumes";
    input.bbox = {{-inp.radius, -inp.radius, -inp.radius},
                  {inp.radius, inp.radius, inp.radius}};

    input.surfaces = {Sphere({0, 0, 0}, inp.radius)};
    input.surface_labels = {Label("sphere")};
    input.volumes = [&input] {
        std::vector<VolumeInput> result;
        VolumeInput vi;
        vi.faces = {LocalSurfaceId{0}};
        vi.zorder = ZOrder::media;

        // Outside
        vi.logic = {0};
        vi.label = "outside";
        vi.bbox = BBox::from_infinite();
        result.push_back(vi);

        // Inside
        vi.logic = {0, logic::lnot};
        vi.label = "inside";
        vi.bbox = input.bbox;
        result.push_back(vi);
        return result;
    }();

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
        host_state_ = HostStateStore(this->host_params(), 1);
    }
    return host_state_.ref();
}

//---------------------------------------------------------------------------//
/*!
 * Access the params data on the host.
 */
auto OrangeGeoTestBase::host_params() const -> HostParamsRef const&
{
    CELER_EXPECT(params_);
    return params_->host_ref();
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
    auto const& host_ref = this->host_params();
    CELER_ASSERT(host_ref.simple_units.size() == 1);
    LocalSurfaceVisitor visit{host_ref, SimpleUnitId{0}};

    os << "# Surfaces\n";

    // Loop over all surfaces and apply
    for (auto id : range(LocalSurfaceId{this->params().num_surfaces()}))
    {
        os << " - " << this->id_to_label(UniverseId{0}, id) << "(" << id.get()
           << "): ";
        visit([&os](auto const& surf) { os << surf; }, id);
        os << '\n';
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return the number of volumes.
 */
VolumeId::size_type OrangeGeoTestBase::num_volumes() const
{
    CELER_EXPECT(params_);
    return params_->num_volumes();
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

    detail::UniverseIndexer ui(this->params().host_ref().universe_indexer_data);
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

    detail::UniverseIndexer ui(this->params().host_ref().universe_indexer_data);
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

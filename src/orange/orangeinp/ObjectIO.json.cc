//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectIO.json.cc
//---------------------------------------------------------------------------//
#include "ObjectIO.json.hh"

#include <memory>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/cont/SpanIO.json.hh"
#include "corecel/io/JsonPimpl.hh"

#include "CsgObject.hh"
#include "IntersectRegion.hh"
#include "ObjectInterface.hh"
#include "PolySolid.hh"
#include "Shape.hh"
#include "Solid.hh"
#include "Transformed.hh"

#define SIO_ATTR_PAIR(OBJ, ATTR) {#ATTR, OBJ.ATTR()}

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
//! Get a user-facing string for a Joined type
char const* to_type_str(OperatorToken Op)
{
    return Op == op_and ? "all" : Op == op_or ? "any" : "<error>";
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get a JSON string representing an object.
 *
 * \see ObjectInterface.hh
 */
std::string to_string(ObjectInterface const& obj)
{
    JsonPimpl json_wrap;
    obj.output(&json_wrap);
    return json_wrap.obj.dump();
}

//---------------------------------------------------------------------------//
//!@{
//! Write objects to JSON
template<OperatorToken Op>
void to_json(nlohmann::json& j, JoinObjects<Op> const& obj)
{
    j = {{"_type", to_type_str(Op)},
         SIO_ATTR_PAIR(obj, label),
         SIO_ATTR_PAIR(obj, daughters)};
}
template void to_json(nlohmann::json&, JoinObjects<op_and> const&);
template void to_json(nlohmann::json&, JoinObjects<op_or> const&);

void to_json(nlohmann::json& j, NegatedObject const& obj)
{
    j = {{"_type", "negated"},
         SIO_ATTR_PAIR(obj, label),
         SIO_ATTR_PAIR(obj, daughter)};
}

void to_json(nlohmann::json& j, PolyCone const& obj)
{
    j = {{"_type", "polycone"},
         SIO_ATTR_PAIR(obj, label),
         SIO_ATTR_PAIR(obj, segments)};
    if (auto sea = obj.enclosed_angle())
    {
        j["enclosed_angle"] = sea;
    }
}

void to_json(nlohmann::json& j, PolyPrism const& obj)
{
    j = {
        {"_type", "polyprism"},
        SIO_ATTR_PAIR(obj, label),
        SIO_ATTR_PAIR(obj, segments),
        SIO_ATTR_PAIR(obj, num_sides),
        SIO_ATTR_PAIR(obj, orientation),
    };
    if (auto sea = obj.enclosed_angle())
    {
        j["enclosed_angle"] = sea;
    }
}

void to_json(nlohmann::json& j, ShapeBase const& obj)
{
    j = {{"_type", "shape"},
         SIO_ATTR_PAIR(obj, label),
         SIO_ATTR_PAIR(obj, interior)};
}

void to_json(nlohmann::json& j, SolidBase const& obj)
{
    j = {{"_type", "solid"},
         SIO_ATTR_PAIR(obj, label),
         SIO_ATTR_PAIR(obj, interior)};
    if (auto* cr = obj.excluded())
    {
        j["excluded"] = *cr;
    }
    if (auto sea = obj.enclosed_angle())
    {
        j["enclosed_angle"] = sea;
    }
    if (auto szs = obj.z_slab())
    {
        j["z_slab"] = szs;
    }
}

void to_json(nlohmann::json& j, Transformed const& obj)
{
    j = {{"_type", "transformed"},
         /* no label needed */
         SIO_ATTR_PAIR(obj, daughter),
         SIO_ATTR_PAIR(obj, transform)};
}
//!@}

//---------------------------------------------------------------------------//
//!@{
//! Write helper classes to JSON
void to_json(nlohmann::json& j, PolySegments const& ps)
{
    j = {{SIO_ATTR_PAIR(ps, outer), SIO_ATTR_PAIR(ps, z)}};
    if (ps.has_exclusion())
    {
        j.push_back(SIO_ATTR_PAIR(ps, inner));
    }
}

void to_json(nlohmann::json& j, SolidEnclosedAngle const& sea)
{
    j = {{"start", sea.start().value()}, {"interior", sea.interior().value()}};
}

void to_json(nlohmann::json& j, SolidZSlab const& szs)
{
    j = {{"lower", szs.lower()}, {"upper", szs.upper()}};
}

//---------------------------------------------------------------------------//
//!@{
//! Write intersect regions to JSON
void to_json(nlohmann::json& j, IntersectRegionInterface const& cr)
{
    celeritas::JsonPimpl json_wrap;
    cr.output(&json_wrap);
    j = std::move(json_wrap.obj);
}

void to_json(nlohmann::json& j, Box const& cr)
{
    j = {{"_type", "box"}, SIO_ATTR_PAIR(cr, halfwidths)};
}
void to_json(nlohmann::json& j, Cone const& cr)
{
    j = {{"_type", "cone"},
         SIO_ATTR_PAIR(cr, radii),
         SIO_ATTR_PAIR(cr, halfheight)};
}
void to_json(nlohmann::json& j, Cylinder const& cr)
{
    j = {{"_type", "cylinder"},
         SIO_ATTR_PAIR(cr, radius),
         SIO_ATTR_PAIR(cr, halfheight)};
}
void to_json(nlohmann::json& j, Ellipsoid const& cr)
{
    j = {{"_type", "ellipsoid"}, SIO_ATTR_PAIR(cr, radii)};
}
void to_json(nlohmann::json& j, EllipticalCylinder const& cr)
{
    j = {{"_type", "ellipticalcylinder"},
         SIO_ATTR_PAIR(cr, radii),
         SIO_ATTR_PAIR(cr, halfheight)};
}
void to_json(nlohmann::json& j, EllipticalCone const& cr)
{
    j = {{"_type", "ellipticalcone"},
         SIO_ATTR_PAIR(cr, lower_radii),
         SIO_ATTR_PAIR(cr, upper_radii),
         SIO_ATTR_PAIR(cr, halfheight)};
}
void to_json(nlohmann::json& j, ExtrudedPolygon const& cr)
{
    j = {{"_type", "extrudedpolygon"},
         SIO_ATTR_PAIR(cr, polygon),
         SIO_ATTR_PAIR(cr, bot_line_segment_point),
         SIO_ATTR_PAIR(cr, top_line_segment_point),
         SIO_ATTR_PAIR(cr, bot_scaling_factor),
         SIO_ATTR_PAIR(cr, top_scaling_factor)};
}
void to_json(nlohmann::json& j, GenPrism const& cr)
{
    j = {{"_type", "genprism"},
         SIO_ATTR_PAIR(cr, halfheight),
         SIO_ATTR_PAIR(cr, lower),
         SIO_ATTR_PAIR(cr, upper)};
}
void to_json(nlohmann::json& j, InfSlab const& cr)
{
    j = {{"_type", "infslab"}, {"lower", cr.lower()}, {"upper", cr.upper()}};
}
void to_json(nlohmann::json& j, InfWedge const& cr)
{
    j = {{"_type", "infwedge"},
         {"start", cr.start().value()},
         {"interior", cr.interior().value()}};
}
void to_json(nlohmann::json& j, Parallelepiped const& cr)
{
    j = {{"_type", "parallelepiped"},
         SIO_ATTR_PAIR(cr, halfedges),
         {"alpha", cr.alpha().value()},
         {"theta", cr.theta().value()},
         {"phi", cr.phi().value()}};
}
void to_json(nlohmann::json& j, Prism const& cr)
{
    j = {{"_type", "prism"},
         SIO_ATTR_PAIR(cr, num_sides),
         SIO_ATTR_PAIR(cr, apothem),
         SIO_ATTR_PAIR(cr, halfheight),
         SIO_ATTR_PAIR(cr, orientation)};
}
void to_json(nlohmann::json& j, Sphere const& cr)
{
    j = {{"_type", "sphere"}, SIO_ATTR_PAIR(cr, radius)};
}
void to_json(nlohmann::json& j, Involute const& cr)
{
    j = {{"_type", "involute"},
         SIO_ATTR_PAIR(cr, radii),
         SIO_ATTR_PAIR(cr, displacement_angle),
         SIO_ATTR_PAIR(cr, t_bounds),
         SIO_ATTR_PAIR(cr, chirality),
         SIO_ATTR_PAIR(cr, halfheight)};
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

namespace nlohmann
{
//---------------------------------------------------------------------------//
void adl_serializer<CelerSPObjConst>::to_json(json& j,
                                              CelerSPObjConst const& oi)
{
    j = oi ? celeritas::json_pimpl_output(*oi) : json(nullptr);
}

void adl_serializer<CelerVarTransform>::to_json(json& j,
                                                CelerVarTransform const& vt)
{
    std::visit(
        [&j](auto&& tr) {
            j = {{"_type", to_cstring(tr.transform_type())},
                 {"data", tr.data()}};
        },
        vt);
}

//---------------------------------------------------------------------------//
}  // namespace nlohmann

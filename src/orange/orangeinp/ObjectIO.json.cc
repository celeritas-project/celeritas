//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectIO.json.cc
//---------------------------------------------------------------------------//
#include "ObjectIO.json.hh"

#include <memory>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/cont/SpanIO.json.hh"
#include "corecel/io/JsonPimpl.hh"

#include "ConvexRegion.hh"
#include "CsgObject.hh"
#include "ObjectInterface.hh"
#include "Shape.hh"
#include "Solid.hh"
#include "Transformed.hh"

#define SIO_ATTR_PAIR(OBJ, ATTR) \
    {                            \
        #ATTR, OBJ.ATTR()        \
    }

namespace nlohmann
{
//---------------------------------------------------------------------------//
// Support serialization of shared pointers to ORANGE objects
using SPObjConst = std::shared_ptr<celeritas::orangeinp::ObjectInterface const>;
using VarTransform = celeritas::VariantTransform;

template<>
struct adl_serializer<SPObjConst>
{
    static void to_json(json& j, SPObjConst const& oi)
    {
        if (oi)
        {
            celeritas::JsonPimpl json_wrap;
            oi->output(&json_wrap);
            j = std::move(json_wrap.obj);
        }
        else
        {
            j = nullptr;
        }
    }
};

template<>
struct adl_serializer<VarTransform>
{
    static void to_json(json& j, VarTransform const& vt)
    {
        std::visit(
            [&j](auto&& tr) {
                j = {{"_type", to_cstring(tr.transform_type())},
                     {"data", tr.data()}};
            },
            vt);
    }
};

//---------------------------------------------------------------------------//
}  // namespace nlohmann

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
void to_json(nlohmann::json& j, SolidEnclosedAngle const& sea)
{
    j = {{"start", sea.start().value()}, {"interior", sea.interior().value()}};
}

//---------------------------------------------------------------------------//
//!@{
//! Write convex regions to JSON
void to_json(nlohmann::json& j, ConvexRegionInterface const& cr)
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
void to_json(nlohmann::json& j, InfWedge const& cr)
{
    j = {{"_type", "infwedge"},
         {"start", cr.start().value()},
         {"interior", cr.interior().value()}};
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
//!@}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

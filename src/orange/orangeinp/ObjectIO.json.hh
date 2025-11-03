//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "orange/orangeinp/IntersectRegion.hh"
#include "orange/transform/VariantTransform.hh"

#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
class ObjectInterface;

template<OperatorToken Op>
class JoinObjects;
class NegatedObject;
class PolyCone;
class PolyPrism;
class RevolvedPolygon;
class ShapeBase;
class SolidBase;
class StackedExtrudedPolygon;
class Transformed;
class Truncated;

class PolySegments;
class EnclosedAzi;
class EnclosedPolar;

class IntersectRegionInterface;
class Box;
class Cone;
class CutCylinder;
class Cylinder;
class Ellipsoid;
class EllipticalCone;
class EllipticalCylinder;
class ExtrudedPolygon;
class GenPrism;
class InfPlane;
class InfAziWedge;
class InfPolarWedge;
class Involute;
class Paraboloid;
class Parallelepiped;
class Prism;
class Sphere;
class Tet;

//---------------------------------------------------------------------------//

// Dump an object to a string
std::string to_string(ObjectInterface const&);

// Write objects to JSON
template<OperatorToken Op>
void to_json(nlohmann::json& j, JoinObjects<Op> const&);
void to_json(nlohmann::json& j, NegatedObject const&);
void to_json(nlohmann::json& j, PolyCone const&);
void to_json(nlohmann::json& j, PolyPrism const&);
void to_json(nlohmann::json& j, RevolvedPolygon const&);
void to_json(nlohmann::json& j, ShapeBase const&);
void to_json(nlohmann::json& j, SolidBase const&);
void to_json(nlohmann::json& j, StackedExtrudedPolygon const&);
void to_json(nlohmann::json& j, Transformed const&);
void to_json(nlohmann::json& j, Truncated const& tr);

// Write helper classes to JSON
void to_json(nlohmann::json& j, PolySegments const&);
void to_json(nlohmann::json& j, EnclosedAzi const&);
void to_json(nlohmann::json& j, EnclosedPolar const&);

// Write intersect regions to JSON
void to_json(nlohmann::json& j, IntersectRegionInterface const&);
void to_json(nlohmann::json& j, Box const&);
void to_json(nlohmann::json& j, Cone const&);
void to_json(nlohmann::json& j, CutCylinder const&);
void to_json(nlohmann::json& j, Cylinder const&);
void to_json(nlohmann::json& j, Ellipsoid const&);
void to_json(nlohmann::json& j, EllipticalCone const&);
void to_json(nlohmann::json& j, EllipticalCylinder const&);
void to_json(nlohmann::json& j, ExtrudedPolygon const&);
void to_json(nlohmann::json& j, GenPrism const&);
void to_json(nlohmann::json& j, Hyperboloid const&);
void to_json(nlohmann::json& j, InfPlane const& a);
void to_json(nlohmann::json& j, InfAziWedge const&);
void to_json(nlohmann::json& j, InfPolarWedge const&);
void to_json(nlohmann::json& j, Involute const&);
void to_json(nlohmann::json& j, Paraboloid const&);
void to_json(nlohmann::json& j, Parallelepiped const&);
void to_json(nlohmann::json& j, Prism const&);
void to_json(nlohmann::json& j, Sphere const&);
void to_json(nlohmann::json& j, Tet const&);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

namespace nlohmann
{
//---------------------------------------------------------------------------//
// Support serialization of shared pointers to ORANGE objects
using CelerSPObjConst
    = std::shared_ptr<celeritas::orangeinp::ObjectInterface const>;
using CelerVarTransform = celeritas::VariantTransform;

template<>
struct adl_serializer<CelerSPObjConst>
{
    static void to_json(json& j, CelerSPObjConst const& oi);
};

template<>
struct adl_serializer<CelerVarTransform>
{
    static void to_json(json& j, CelerVarTransform const& vt);
};

//---------------------------------------------------------------------------//
}  // namespace nlohmann

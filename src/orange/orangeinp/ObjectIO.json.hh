//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <nlohmann/json.hpp>

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
class ShapeBase;
class SolidBase;
class Transformed;

class PolySegments;
class SolidEnclosedAngle;
class SolidZSlab;

class IntersectRegionInterface;
class Box;
class Cone;
class Cylinder;
class Ellipsoid;
class EllipticalCylinder;
class EllipticalCone;
class ExtrudedPolygon;
class GenPrism;
class InfSlab;
class InfWedge;
class Parallelepiped;
class Prism;
class Sphere;
class Involute;

//---------------------------------------------------------------------------//

// Dump an object to a string
std::string to_string(ObjectInterface const&);

// Write objects to JSON
template<OperatorToken Op>
void to_json(nlohmann::json& j, JoinObjects<Op> const&);
void to_json(nlohmann::json& j, NegatedObject const&);
void to_json(nlohmann::json& j, PolyCone const&);
void to_json(nlohmann::json& j, PolyPrism const&);
void to_json(nlohmann::json& j, ShapeBase const&);
void to_json(nlohmann::json& j, SolidBase const&);
void to_json(nlohmann::json& j, Transformed const&);

// Write helper classes to JSON
void to_json(nlohmann::json& j, PolySegments const&);
void to_json(nlohmann::json& j, SolidEnclosedAngle const&);
void to_json(nlohmann::json& j, SolidZSlab const&);

// Write intersect regions to JSON
void to_json(nlohmann::json& j, IntersectRegionInterface const& cr);
void to_json(nlohmann::json& j, Box const& cr);
void to_json(nlohmann::json& j, Cone const& cr);
void to_json(nlohmann::json& j, Cylinder const& cr);
void to_json(nlohmann::json& j, Ellipsoid const& cr);
void to_json(nlohmann::json& j, EllipticalCylinder const& cr);
void to_json(nlohmann::json& j, EllipticalCone const& cr);
void to_json(nlohmann::json& j, ExtrudedPolygon const& cr);
void to_json(nlohmann::json& j, GenPrism const& cr);
void to_json(nlohmann::json& j, InfSlab const& cr);
void to_json(nlohmann::json& j, InfWedge const& cr);
void to_json(nlohmann::json& j, Parallelepiped const& cr);
void to_json(nlohmann::json& j, Prism const& cr);
void to_json(nlohmann::json& j, Sphere const& cr);
void to_json(nlohmann::json& j, Involute const& cr);

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

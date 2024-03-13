//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ObjectIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

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
class ShapeBase;
class SolidBase;
class Transformed;

class SolidEnclosedAngle;

class ConvexRegionInterface;
class Box;
class Cone;
class Cylinder;
class Ellipsoid;
class InfWedge;
class Prism;
class Sphere;

//---------------------------------------------------------------------------//

// Dump an object to a string
std::string to_string(ObjectInterface const&);

// Write objects to JSON
template<OperatorToken Op>
void to_json(nlohmann::json& j, JoinObjects<Op> const& sb);
void to_json(nlohmann::json& j, NegatedObject const& sb);
void to_json(nlohmann::json& j, ShapeBase const& sb);
void to_json(nlohmann::json& j, SolidBase const& sb);
void to_json(nlohmann::json& j, Transformed const& sb);

// Write helper classes to JSON
void to_json(nlohmann::json& j, SolidEnclosedAngle const& sea);

// Write convex regions to JSON
void to_json(nlohmann::json& j, ConvexRegionInterface const& cr);
void to_json(nlohmann::json& j, Box const& cr);
void to_json(nlohmann::json& j, Cone const& cr);
void to_json(nlohmann::json& j, Cylinder const& cr);
void to_json(nlohmann::json& j, Ellipsoid const& cr);
void to_json(nlohmann::json& j, InfWedge const& cr);
void to_json(nlohmann::json& j, Prism const& cr);
void to_json(nlohmann::json& j, Sphere const& cr);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

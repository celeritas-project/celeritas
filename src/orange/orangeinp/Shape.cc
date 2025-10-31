//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Shape.cc
//---------------------------------------------------------------------------//
#include "Shape.hh"

#include "corecel/io/JsonPimpl.hh"

#include "ObjectIO.json.hh"

#include "detail/BuildIntersectRegion.hh"

namespace celeritas
{
namespace orangeinp
{

//---------------------------------------------------------------------------//
/*!
 * Construct with label from a daughter class.
 */
ShapeBase::ShapeBase(std::string&& label) : label_{std::move(label)}
{
    CELER_EXPECT(!label_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId ShapeBase::build(VolumeBuilder& vb) const
{
    return detail::build_intersect_region(
        vb, this->label(), {}, this->interior());
}

//---------------------------------------------------------------------------//
/*!
 * Output to JSON.
 */
void ShapeBase::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Shape<Box>;
template class Shape<Cone>;
template class Shape<CutCylinder>;
template class Shape<Cylinder>;
template class Shape<Ellipsoid>;
template class Shape<ExtrudedPolygon>;
template class Shape<GenPrism>;
template class Shape<Involute>;
template class Shape<Parallelepiped>;
template class Shape<Prism>;
template class Shape<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

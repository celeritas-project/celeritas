//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
template class Shape<Cylinder>;
template class Shape<Ellipsoid>;
template class Shape<GenPrism>;
template class Shape<Parallelepiped>;
template class Shape<Prism>;
template class Shape<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

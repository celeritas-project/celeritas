//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Shape.cc
//---------------------------------------------------------------------------//
#include "Shape.hh"

#include "corecel/io/JsonPimpl.hh"

#include "detail/BuildConvexRegion.hh"

#if CELERITAS_USE_JSON
#    include "ObjectIO.json.hh"
#endif

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
    return detail::build_convex_region(
        vb, std::string{this->label()}, {}, this->interior());
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
template class Shape<Prism>;
template class Shape<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/FaceNamer.cc
//---------------------------------------------------------------------------//
#include "FaceNamer.hh"

#include "corecel/Assert.hh"

#include "VariantSurface.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
constexpr char to_pm(Sense s)
{
    return s == Sense::inside ? 'p' : 'm';
}

#define ORANGE_INSTANTIATE_OP(IN)                                        \
    template std::string FaceNamer::Impl::operator()(IN<Axis::x> const&) \
        const;                                                           \
    template std::string FaceNamer::Impl::operator()(IN<Axis::y> const&) \
        const;                                                           \
    template std::string FaceNamer::Impl::operator()(IN<Axis::z> const&) const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Apply to a surface with unknown type.
 */
std::string FaceNamer::operator()(Sense s, VariantSurface const& surf)
{
    CELER_ASSUME(!surf.valueless_by_exception());
    return std::visit(Impl{&state_, s}, surf);
}

//---------------------------------------------------------------------------//
// IMPL DEFINITIONS
//---------------------------------------------------------------------------//
template<Axis T>
std::string FaceNamer::Impl::operator()(PlaneAligned<T> const&) const
{
    return {to_pm(sense_), to_char(T)};
}

//! \cond
ORANGE_INSTANTIATE_OP(PlaneAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a name for an along-axis cylinder.
 */
template<Axis T>
std::string FaceNamer::Impl::operator()(CylCentered<T> const&) const
{
    return {'c', to_char(T)};
}

//! \cond
ORANGE_INSTANTIATE_OP(CylCentered);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a centered sphere.
 */
std::string FaceNamer::Impl::operator()(SphereCentered const&) const
{
    return "s";
}

//---------------------------------------------------------------------------//
/*!
 * Construct a name for an axis-aligned cylinder.
 */
template<Axis T>
std::string FaceNamer::Impl::operator()(CylAligned<T> const&) const
{
    return {'c', to_char(T)};
}

//! \cond
ORANGE_INSTANTIATE_OP(CylAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a general plane.
 */
std::string FaceNamer::Impl::operator()(Plane const&) const
{
    return "p" + std::to_string(state_->num_planes_++);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a sphere.
 */
std::string FaceNamer::Impl::operator()(Sphere const&) const
{
    return "s";
}

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a cone.
 */
template<Axis T>
std::string FaceNamer::Impl::operator()(ConeAligned<T> const&) const
{
    return {'k', to_char(T)};
}

//! \cond
ORANGE_INSTANTIATE_OP(ConeAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a simple quadric.
 */
std::string FaceNamer::Impl::operator()(SimpleQuadric const&) const
{
    return "sq";
}

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a quadric.
 */
std::string FaceNamer::Impl::operator()(GeneralQuadric const&) const
{
    return "gq";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

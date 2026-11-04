//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#define ORANGE_INSTANTIATE_OP(IN) \
    template std::string FaceNamer::Impl::operator()(IN<Axis::x> const&) \
        const; \
    template std::string FaceNamer::Impl::operator()(IN<Axis::y> const&) \
        const; \
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
    std::string result = prefix_;
    result += std::visit(Impl{&state_, s}, surf);
    return result;
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
 * Construct a name for a general plane.
 */
std::string FaceNamer::Impl::operator()(Plane const&) const
{
    return "p" + std::to_string(state_->num_plane++);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a name for a general quadric.
 *
 * Although some shapes are a single GQ, others (e.g. generic trapezoid) can
 * have faces comprised of GQs.
 */
std::string FaceNamer::Impl::operator()(GeneralQuadric const&) const
{
    std::string result{"gq"};
    if (int gqid = state_->num_gq++; gqid > 0)
    {
        result += std::to_string(gqid);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
